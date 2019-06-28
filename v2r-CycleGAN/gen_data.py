import torch
from model import G12, G21
from torchvision import transforms
import numpy as np
import cv2
import os

transform2tensor = transforms.Compose([
    transforms.ToTensor()
    ])

transform2pil = transforms.Compose([
    transforms.ToPILImage()
    ])

count_step = 2000

def gen_data(model, model_path, source_path, target_path):
    Net = model()
    Net.cuda()
    Net.load_state_dict(torch.load(model_path))
    source_data = np.load(source_path)

    
    # Net.eval()
    print('model loaded!')
    np_data_list = []
    for i in range(source_data.shape[0]):
        name = str(i).zfill(5)
        each_img = source_data[i]
        img = cv2.resize(each_img, (160,56), interpolation=cv2.INTER_CUBIC)
        each_tensor = transform2tensor(img)
        each_tensor = torch.unsqueeze(each_tensor, dim=0)
        each_tensor = each_tensor.cuda()
        # print(each_tensor.shape)
        with torch.no_grad():
            out_tensor = Net(each_tensor)
        out_tensor = torch.squeeze(out_tensor, dim=0)
        # out_pil = transform2pil(out_tensor.cpu().detach())
        # print(out_pil.size)
        out_np = out_tensor.cpu().detach().numpy()*255
        out_np = out_np.astype(np.uint8)
        out_np = out_np.squeeze()
        out_np = out_np.transpose(1, 2, 0)
        out_np = cv2.cvtColor(out_np,cv2.COLOR_BGR2RGB)
        out_np = cv2.resize(out_np, (160,50), interpolation=cv2.INTER_CUBIC)
        # print(out_np.shape)
        if i % count_step == 0:
            cv2.imwrite(os.path.join(target_path, '{}.png'.format(name)), out_np)
        # out_pil.save(os.path.join(target_path, '{}.png'.format(name)))
        print(name+' is done!')
        np_data_list.append(out_np)
        

    np_data = np.array(np_data_list)
    print(np_data.shape)
    np.save(os.path.join(target_path, 'train_without_night_im_fake.npy'), np_data)




if __name__ == '__main__':
    model_path = '/home/jinlukang/DailyIssues/issue-Notebook/v2r-CycleGAN/models/Gv2r-5000.pkl'
    source_path = '/data1/jinlukang/LPR/train_without_night_im.npy'
    target_path = '/home/jinlukang/DailyIssues/issue-Notebook/v2r-CycleGAN/gen_data'
    gen_data(G21, model_path, source_path, target_path)