import numpy as np
import sys
import PIL.Image
import PIL.ImageFile
import PIL.ImageFilter

(PIL.ImageFile).LOAD_TRUNCATED_IMAGES = True
NUMBER_OF_CLASSES = 110
SIZE = 299
SIZE_TARGET = 200

InputDirectory = sys.argv[1]
OutputDirectory = sys.argv[2]

File = open(InputDirectory + "/dev.csv")
File.readline()

def img_loader(path):
    with open(path, 'rb') as f:
        with PIL.Image.open(f).convert('RGB') as image:
            image = image.resize((299, 299), PIL.Image.ANTIALIAS)
            image = np.array(image).astype(np.float32)
    return image
itr = 1

def hole_image(image_pil_ori, image_pil_target):
    new_image = image_pil_ori.copy().crop((120,160,140,180))
    temp = image_pil_target.copy()
    temp.paste(new_image, (120,160))
    return temp

total_score = 0.
for line in File:
    split_line = line.strip().split(",")

    filename = split_line[0]
    # true_label = int(split_line[1])
    targeted_label = int(split_line[2])

    File2 = open(InputDirectory + "/dev.csv")
    File2.readline()

    raw_image = img_loader(InputDirectory + "/" + filename)

    score = 1000.
    for line2 in File2:
        split_line2 = line2.strip().split(",")

        target_filename = split_line2[0]
        true_label = int(split_line2[1])

        if targeted_label == true_label:
            # comp_name = target_filename
            comp_image = img_loader(InputDirectory + "/" + target_filename)
            comp_score = np.sqrt(np.mean((raw_image-comp_image)**2))


            if comp_score < score:
                score = comp_score
                target_name = target_filename
            print(score)
    
    File2.close()

    image_target = PIL.Image.open(InputDirectory + "/" + target_name).convert("RGB")
    image_origin = PIL.Image.open(InputDirectory + "/" + filename).convert("RGB")

    # print('{},{}'.format(np.mean(np.asarray(image_target)[:,:,0]), np.mean(np.asarray(image_origin)[:,:,0])))

    image_target_np = np.asarray(image_target).astype(np.float32)
    # print("image_target_np:{}".format(image_target_np.shape))
    image_origin_np = np.asarray(image_origin).astype(np.float32)
    # print("image_origin_np:{}".format(image_origin_np.shape))

    image_target = PIL.Image.fromarray(np.uint8(image_target_np))
    
    image_target = image_target.resize((SIZE, SIZE),PIL.Image.ANTIALIAS)
    image_origin = image_origin.resize((SIZE, SIZE),PIL.Image.ANTIALIAS)


    blend_image = PIL.Image.blend(image_origin, image_target, 1.0)
    temp_target = blend_image.copy()
    crop_image = temp_target.crop((50, 50, 250, 250))
    a = 1
    # print(np.asarray(fin_image_target))
    image_fin = image_origin.copy()
    # print(temp_image)
    image_fin.paste(crop_image,(50, 50))

    image_fin_np = np.asarray(image_fin).astype(np.float32)
    # print(fin_img_np)

    fin_score = np.sqrt(np.mean((image_fin_np-image_origin_np)**2))
    
    if fin_score >= 64:
        image_fin = image_origin.copy()
        target_fin = blend_image.copy().crop((75, 75, 225, 225))
        image_fin.paste(target_fin,(75, 75))
        fin_score = np.sqrt(np.mean((np.asarray(image_fin).astype(np.float32)-image_origin_np)**2))
        a = 2
        if fin_score >= 64:
            image_fin = image_origin.copy()
            target_fin = blend_image.copy().crop((100, 100, 200, 200))
            image_fin.paste(target_fin,(100, 100))
            fin_score = np.sqrt(np.mean((np.asarray(image_fin).astype(np.float32)-image_origin_np)**2))
            a = 3
    total_score += fin_score
    print('name:{}, fin_score:{}, mode:{}'.format(filename, fin_score, a))
    itr += 1

    new_image = image_fin.copy()
    fin_new_image = hole_image(image_origin, new_image)
    fin_new_image.save(OutputDirectory + "/" + filename)

print('mean_score:{}'.format(total_score/110))
File.close()