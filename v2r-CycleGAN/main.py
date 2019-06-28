import torch
import argparse
import os, sys
from torch.utils.data import DataLoader
import numpy as np
import PIL
import cv2
import torchvision.utils as vutils
from torchvision import transforms
from DataLoader import License_Real, License_Virtual
from tensorboardX import SummaryWriter
from model import G12, G21
from model import D1, D2

transform2tensor = transforms.Compose([
    transforms.ToTensor()
    ])

transform2pil = transforms.Compose([
    transforms.ToPILImage()
    ])

def merge_images(config, sources, targets):
    _, _, h, w = sources.shape
    # print(sources.shape)
    row = 8
    s_t_list = []
    for i, (s, t) in enumerate(zip(sources, targets)):
        s = s.transpose(1, 2, 0)
        t = t.transpose(1, 2, 0)
        s_t = np.hstack((s, t))
        s_t_list.append(s_t)
    s_t_c1 = np.vstack(tuple(s_t_list[:8]))
    s_t_c2 = np.vstack(tuple(s_t_list[8:16]))
    s_t_c3 = np.vstack(tuple(s_t_list[16:24]))
    s_t_c4 = np.vstack(tuple(s_t_list[24:32]))
    merged = np.hstack((s_t_c1, s_t_c2, s_t_c3, s_t_c4))
    return merged#cv2.cvtColor(merged,cv2.COLOR_BGR2RGB)#.transpose(1, 2, 0)

    
def train(config):

    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    #load data
    Real_Dataset = License_Real(config.real_path)
    Real_Dataloader = DataLoader(Real_Dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)

    Virtual_Dataset = License_Virtual(config.virtual_path)
    Virtual_Dataloader = DataLoader(Virtual_Dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)

    #model and optim
    Gv2r = G12(conv_dim=config.g_conv_dim)
    Gr2v = G21(conv_dim=config.g_conv_dim)
    Dv = D1(conv_dim=config.d_conv_dim)
    Dr = D2(conv_dim=config.d_conv_dim)

    g_params = list(Gv2r.parameters()) + list(Gr2v.parameters())
    d_params = list(Dv.parameters()) + list(Dr.parameters())

    g_optimizer = torch.optim.Adam(g_params, config.lr, [config.beta1, config.beta2])
    d_optimizer = torch.optim.Adam(d_params, config.lr, [config.beta1, config.beta2])

    if torch.cuda.is_available():
        Gv2r.cuda()
        Gr2v.cuda()
        Dv.cuda()
        Dr.cuda()

    Real_iter = iter(Real_Dataloader)
    Virtual_iter = iter(Virtual_Dataloader)

    Real_sample_batch = Real_iter.next()
    Virtual_sample_batch = Virtual_iter.next()

    Real_sample_batch = Real_sample_batch.cuda()
    Virtual_sample_batch = Virtual_sample_batch.cuda()
    # print(Real_sample_batch.shape)
    #train
    step = 0

    tb_logger = SummaryWriter('./logs')

    for each_epoch in range(config.train_epochs):
        for itr, (r_batch_data, v_batch_data) in enumerate(zip(Real_Dataloader, Virtual_Dataloader)):
            #============ train D ============#
            # train with real images
            r_batch_data = r_batch_data.cuda()
            v_batch_data = v_batch_data.cuda()
            # print(r_batch_data.shape)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            out = Dr(r_batch_data)
            dr_loss = torch.mean((out-1)**2)

            out = Dv(v_batch_data)
            dv_loss = torch.mean((out-1)**2)

            d_real_loss = dr_loss + dv_loss
            d_real_loss.backward()
            d_optimizer.step()

            # train with fake images
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            fake_v = Gr2v(r_batch_data)
            out = Dv(fake_v)

            dv_loss = torch.mean(out**2)

            fake_r = Gv2r(v_batch_data)
            out = Dr(fake_r)

            dr_loss = torch.mean(out**2)

            d_fake_loss = dv_loss + dr_loss
            d_fake_loss.backward()
            d_optimizer.step()

            #============ train G ============#
            # train r-v-r cycle
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            fake_v = Gr2v(r_batch_data)
            out = Dv(fake_v)
            reconst_r = Gv2r(fake_v)
            # print(reconst_r.shape)

            g_loss = torch.mean((out-1)**2) + torch.mean((r_batch_data - reconst_r)**2)
            g_loss.backward()
            g_optimizer.step()

            # train v-r-v cycle
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            fake_r = Gv2r(v_batch_data)
            out = Dr(fake_r)
            reconst_v = Gr2v(fake_r)

            g_loss = torch.mean((out-1)**2) + torch.mean((v_batch_data - reconst_v)**2)
            g_loss.backward()
            g_optimizer.step()

            #print the log
            tb_logger.add_scalar('d_real_loss', d_real_loss, step)
            tb_logger.add_scalar('d_fake_loss', d_fake_loss, step)
            tb_logger.add_scalar('g_loss', g_loss, step)
            tb_logger.add_scalar('dv_loss', dv_loss, step)
            tb_logger.add_scalar('dr_loss', dr_loss, step)

            print('step:{}, d_real_loss:{}, d_fake_loss:{}, g_loss:{}, dv_loss:{}, dr_loss:{}'.format(
                step, d_real_loss, d_fake_loss, g_loss, dv_loss, dr_loss))

            #save the sampled image
            if (step+1) % config.sample_step == 0:
                fake_v = Gv2r(Virtual_sample_batch)
                fake_r = Gr2v(Real_sample_batch)

                fake_r_np = fake_r.cpu().detach().numpy()*255
                fake_v_np = fake_v.cpu().detach().numpy()*255

                real_r_np = Real_sample_batch.cpu().detach().numpy()*255
                real_v_np = Virtual_sample_batch.cpu().detach().numpy()*255
                # print(real_r_np.shape, real_v_np.shape)

                r_merged_image = merge_images(config, real_r_np, fake_r_np)   
                v_merged_image = merge_images(config, real_v_np, fake_v_np)
                r_sample = r_merged_image.copy()
                v_sample = v_merged_image.copy()
                r_merged_image = transform2tensor(r_merged_image)
                v_merged_image = transform2tensor(v_merged_image)
                x1 = vutils.make_grid(r_merged_image, normalize=True, scale_each=True)
                x2 = vutils.make_grid(v_merged_image, normalize=True, scale_each=True)

                tb_logger.add_image('r_Imgs', x1, step+1)
                tb_logger.add_image('v_Imgs', x2, step+1)
                # print(r_merged_image.shape, v_merged_image.shape)
                # save sample

            if (step+1) % config.save_step == 0:
                Gv2r_path = os.path.join(config.model_path, 'Gv2r-{}.pkl'.format(step+1))
                Gr2v_path = os.path.join(config.model_path, 'Gr2v-{}.pkl'.format(step+1))
                Dr_path = os.path.join(config.model_path, 'Dr-{}.pkl'.format(step+1))
                Dv_path = os.path.join(config.model_path, 'Dv-{}.pkl'.format(step+1))
                torch.save(Gv2r.state_dict(), Gv2r_path)
                torch.save(Gr2v.state_dict(), Gr2v_path)
                torch.save(Dr.state_dict(), Dr_path)
                torch.save(Dv.state_dict(), Dv_path)

                cv2.imwrite(os.path.join(config.sample_path, 'r_sample_{}.png'.format(str(step+1).zfill(5))), cv2.cvtColor(r_sample,cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(config.sample_path, 'v_sample_{}.png'.format(str(step+1).zfill(5))), cv2.cvtColor(v_sample,cv2.COLOR_BGR2RGB))

        #     r_np_data = r_batch_data.cpu().detach().numpy()
        #     v_np_data = v_batch_data.cpu().detach().numpy()
        #     merged_image = merge_images(config, r_np_data, v_np_data)
        #     cv2.imwrite(os.path.join(config.sample_path, 'sample_{}.png'.format(str(step).zfill(5))), merged_image)
            step += 1





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    
    # training hyper-parameters
    parser.add_argument('--train_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # misc
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--real_path', type=str, default='/data1/jinlukang/LPR/real_train_im.npy')
    parser.add_argument('--virtual_path', type=str, default='/data1/jinlukang/LPR/train_without_night_im.npy')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=200)
    parser.add_argument('--save_step', type=int , default=5000)

    config = parser.parse_args()
    print(config)
    train(config)