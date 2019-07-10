import torch
from model import G12, G21
from torchvision import transforms
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from DataLoader import License_Real, License_Virtual

transform2tensor = transforms.Compose([
    transforms.ToTensor()
    ])

transform2pil = transforms.Compose([
    transforms.ToPILImage()
    ])

count_step = 2000

def gen_data_with_single(model, model_path, source_path, target_path):
    Net = model()
    Net.cuda()
    # Net.train()
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

def gen_data_with_batch(model, model_path, source_path, split, target_path):
    Net = model()
    Net.cuda()
    Net.load_state_dict(torch.load(model_path))
    # source_data = np.load(os.path.join(source_path, split))
    print('model loaded!')

    Virtual_Dataset = License_Virtual(source_path, split)
    Virtual_Dataloader = DataLoader(Virtual_Dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

    step = 0
    np_data_list = []
    for itr, (data_batch, _, _) in enumerate(Virtual_Dataloader):
        data_batch = data_batch.cuda()
        # print(data_batch)
        out_data_batch = Net(data_batch)
        out_data_batch = torch.clamp(out_data_batch, min=0, max=1)
        print(out_data_batch.shape)
        out_np_batch = out_data_batch.cpu().detach().numpy()*255
        out_np_batch = out_np_batch.astype(np.uint8)

        # print(out_np_batch)
        for i in range(out_np_batch.shape[0]):
            each_np_img = out_np_batch[i]
            each_np_img = each_np_img.transpose(1, 2, 0)
            each_np_img = cv2.resize(each_np_img, (160,50), interpolation=cv2.INTER_CUBIC)
            # print(each_np_img.shape)
            if step % count_step == 0:
                cv2.imwrite(os.path.join(target_path, '{}.png'.format(str(step).zfill(5))), cv2.cvtColor(each_np_img,cv2.COLOR_BGR2RGB))
            step += 1
            np_data_list.append(each_np_img)
        # print(len(out_np_list[0][1]))

    np_data = np.array(np_data_list)
    print('the out np file shape is {}'.format(np_data.shape))
    np.save(os.path.join(target_path, 'angle0_without_night_im.npy'), np_data)






if __name__ == '__main__':
    model_path = '/home/jinlukang/DailyIssues/issue-Notebook/v2r-CycleGAN/models/seg1e-2_angle0_halfcycle/Gv2r-seg1e-2_angle0_halfcycle-5000.pkl'
    source_path = '/data1/jinlukang/LPR/'
    split = 'angle0_without_night'
    target_path = '/home/jinlukang/DailyIssues/issue-Notebook/v2r-CycleGAN/gen_data'
    gen_data_with_batch(G21, model_path, source_path, split, target_path)