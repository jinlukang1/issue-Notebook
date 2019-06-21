import sys
# sys.path.insert(0, 'lib')

import PIL.Image
import PIL.ImageFile
import PIL.ImageFilter
import numpy

(PIL.ImageFile).LOAD_TRUNCATED_IMAGES = True
NUMBER_OF_CLASSES = 110
SIZE = 299

InputDirectory = sys.argv[1]
OutputDirectory = sys.argv[2]

File = open(InputDirectory + "/dev.csv")
File.readline()

for line in File:
    split_line = line.strip().split(",")

    filename = split_line[0]
    true_label = int(split_line[1])
    targeted_label = int(split_line[2])

    image_pil = PIL.Image.open(InputDirectory + "/" + filename)
    image = numpy.asarray(image_pil.resize([SIZE, SIZE], PIL.Image.BILINEAR).convert("RGB")).astype(numpy.float32)

    left = 36
    right = 263
    top = 36
    bottom = 263

    new_image = image.copy()
    n = 1
    for i in range(35):
        new_image[left:right, top:bottom] += image[(left - i):(right - i), top:bottom]
        new_image[left:right, top:bottom] += image[(left + i):(right + i), top:bottom]
        new_image[left:right, top:bottom] += image[left:right, (top - i):(bottom - i)]
        new_image[left:right, top:bottom] += image[left:right, (top + i):(bottom + i)]
        n += 4

    new_image[left:right, top:bottom] /= n
    new_image = new_image.astype(numpy.uint8)

    PIL.Image.fromarray(numpy.asarray(new_image, numpy.int8), "RGB").save(OutputDirectory + "/" + filename)

File.close()