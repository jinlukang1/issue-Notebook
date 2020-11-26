# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt

import torch, torch.utils.data, torchvision
from skimage import transform

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change to available GPU device 

import utils
import cv2, copy

device = torch.device("cuda")
dtype = torch.float32

data_path = "/data3/home_jinlukang/pengkunfu/rectified meters v5" # path of cropped rectified meter images
# data_path = "E:\\biaoji_data_v3\\rectified meters v5"

log_file = "{} train log.txt".format(os.path.basename(__file__).split('.')[0])

def get_train_test_records(meter_type):
    train_records = []
    test_records = []
    im_names = os.listdir(os.path.join(data_path,"biaoji_{}".format("ABCDE"[meter_type]),"images")) 
    gt_path = os.path.join(data_path,"biaoji_{}".format("ABCDE"[meter_type]),"groundtruth percents.txt") # path of groundtruth meter reading percents
    with open(gt_path) as f:
        for line in f:
            im_name, percent = line.split('\t')
            if im_name in im_names:
                percent = float(percent)
                im_index = int(im_name.split('.')[0][-5:])
                if im_index < 1500:
                    train_records.append((im_name,percent))
                    pass
                else:
                    test_records.append((im_name,percent))
                    pass
                pass
            pass
        pass
    return train_records, test_records

class PointerMeters(torch.utils.data.Dataset):
    '''
    records: [(im_name, percent), (im_name, percent), ...]
    '''
    def __init__(self,meter_type,records,transform = None):
        self.meter_type = meter_type
        self.records = records
        self.transform = transform
        return
    def __len__(self):
        return len(self.records)
    def __getitem__(self,i):
        fname = self.records[i][0]
        im = plt.imread(os.path.join(data_path,"biaoji_{}".format("ABCDE"[self.meter_type]),"images",fname))[:,:,:3]
        if im.max() > 1:
            im = im / 255
            pass
        theta = utils.theta_ranges[self.meter_type][0] + self.records[i][1] * (utils.theta_ranges[self.meter_type][1] - utils.theta_ranges[self.meter_type][0])
        sample = {"im":im,
                  "cossin":np.array([np.cos(theta),np.sin(theta)])}
        if self.transform:
            sample = self.transform(sample)
        return sample
    pass

class RandomCrop:
    def __init__(self,meter_type):
        self.meter_type = meter_type
        return
    def __call__(self,sample):
        im = sample["im"]
        H, W = im.shape[:2]
        center = (np.array([H / 2, W / 2]) * (1 + np.random.uniform(-0.1,0.1))).astype(np.int)
        D = np.int(np.minimum(H,W // ((2 if self.meter_type == 2 else 1))) * (1 + np.random.uniform(0,0.3)) / 4)
        D = np.min(np.array([D,
                             center[0], H - center[0],
                             center[1] // (2 if self.meter_type == 2 else 1), (W - center[1]) // (2 if self.meter_type == 2 else 1)]))
        im = im[(center[0] - D) : (center[0] + D),
                (center[1] - D * (2 if self.meter_type == 2 else 1)) : (center[1] + D * (2 if self.meter_type == 2 else 1))]
        sample["im"] = im
        return sample
    pass

class Resize:
    '''
    size: the diameter of the meter
    '''
    def __init__(self,meter_type,size = 256):
        self.meter_type = meter_type
        self.size = size
        return
    def __call__(self,sample):
        im = sample["im"]
        im = transform.resize(im,(self.size,self.size * (2 if self.meter_type == 2 else 1)))
        sample["im"] = im
        return sample
    pass

class CentralCrop:
    def __init__(self,meter_type,size = 256):
        self.meter_type = meter_type
        self.size = size
        return
    def __call__(self,sample):
        im = sample["im"]
        H, W = im.shape[:2]
        D = np.minimum(H,W // (2 if self.meter_type == 2 else 1))
        D = np.int((1.5 if self.meter_type == 2 else 1.3) * D)
        D = np.min(np.array([D, 2 * H, 2 * W]))
        im = im[(H // 2 - D // 4):(H // 2 + D // 4),
                (W // 2 - D // (2 if self.meter_type == 2 else 4)):(W // 2 + D // (2 if self.meter_type == 2 else 4))]
        sample["im"] = im
        return sample
    pass

class ToTensor:
    def __call__(self,sample):
        im, cossin = sample["im"],sample["cossin"]
        return {"im":torch.from_numpy(im.transpose([2,0,1])).to(device = device,dtype = dtype),
                "cossin":torch.from_numpy(cossin).to(device = device,dtype = dtype)}
    pass

def show_samples(dataloader,mode,n = 4):
    '''
    '''
    sample = next(iter(dataloader))
    ims, cossins = sample["im"].cpu().numpy().transpose([0,2,3,1]), sample["cossin"].cpu().numpy()
    H, W = ims.shape[1],ims.shape[2]
    im_show = np.ones([H * n, W * n, 3])
    for i in range(n ** 2):
        im = ims[i].copy()
        cv2.putText(im, text = "{:.3f}".format(utils.orient_to_reading(cossins[i],dataloader.dataset.meter_type)),
                    org = (15,15), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.5, color = (1,0,0), thickness = 1)
        r, c = i // n, i % n
        im_show[(r * H):((r + 1) * H), (c * W):((c + 1) * W)] = im
        pass
    plt.imsave("{}_{}_{}_samples.png".format(os.path.basename(__file__).split('.')[0], "ABCDE"[dataloader.dataset.meter_type],mode), im_show)
    return

def get_dataloaders(meter_type, batch_size = 20, trial = False):
    train_records, test_records = get_train_test_records(meter_type)
    with open(log_file, "a") as f:
        f.write("{}: train = {}, test = {}\n".format("ABCDE"[meter_type],len(train_records),len(test_records)))
    
    if meter_type == 2:
        batch_size //= 2
        pass
    train_records, val_records = utils.split_train_val(train_records)

    if trial:
        n = 100
        trainset = PointerMeters(meter_type,train_records[:n], transform = torchvision.transforms.Compose([RandomCrop(meter_type),Resize(meter_type), ToTensor()]))
        valset = PointerMeters(meter_type,val_records[:n], transform = torchvision.transforms.Compose([RandomCrop(meter_type),Resize(meter_type), ToTensor()]))
    else:
        trainset = PointerMeters(meter_type,train_records, transform = torchvision.transforms.Compose([RandomCrop(meter_type),Resize(meter_type), ToTensor()]))
        valset = PointerMeters(meter_type,val_records, transform = torchvision.transforms.Compose([RandomCrop(meter_type),Resize(meter_type), ToTensor()]))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    with open(log_file, "a") as f:
        f.write("train: {}, val: {}\n".format(len(train_loader.dataset), len(val_loader.dataset)))
    show_samples(train_loader,"train")

    if trial:
        np.random.shuffle(test_records)
        testset = PointerMeters(meter_type,test_records[:n], transform = torchvision.transforms.Compose([CentralCrop(meter_type), Resize(meter_type), ToTensor()]))
    else:
        testset = PointerMeters(meter_type,test_records, transform = torchvision.transforms.Compose([CentralCrop(meter_type), Resize(meter_type), ToTensor()]))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    with open(log_file, "a") as f:
        f.write("test: {}\n".format(len(test_loader.dataset)))
    show_samples(test_loader,"test")
    return train_loader, val_loader, test_loader

class ModelOrientResNet50(torch.nn.Module):
    '''
    '''
    def __init__(self,meter_type):
        super(ModelOrientResNet50,self).__init__()
        self.meter_type = meter_type
        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.backbone = torch.nn.Sequential(*list(resnet50.children())[:-2])
        self.pool = torch.nn.MaxPool2d(kernel_size = 2)
        self.conv = torch.nn.Conv2d(in_channels = 2048, out_channels = 1024, kernel_size = 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(1024 * 2 * (4 if self.meter_type == 2 else 2),2)
        return
    def forward(self,x):
        conv = self.backbone(x)
        # conv = self.relu(conv) # truth is ... I forgot the ReLU layer when I first wrote this code, but it seemed not matter much
        conv = self.pool(conv)
        conv = self.conv(conv)
        # conv = self.relu(conv)
        conv = self.pool(conv)
        fc = conv.view([conv.shape[0],-1])
        output = self.fc(fc)
        orient = output / torch.norm(output,dim = 1,keepdim = True)
        return orient
    pass

def evaluate(model,dataloader,eps = 1e-8):
    model.eval()
    meter_type = dataloader.dataset.meter_type
    val_acc = np.zeros(2)
    with torch.no_grad():
        for sample in dataloader:
            cossins = model(sample["im"])
            angle_err = torch.acos(torch.clamp(torch.sum(cossins * sample["cossin"], dim = 1),-1 + eps,1 - eps))
            reading_err = (angle_err.detach().cpu().numpy() *
                           (utils.reading_ranges[meter_type,1] - utils.reading_ranges[meter_type,0]) /
                           (utils.theta_ranges[meter_type,1] - utils.theta_ranges[meter_type,0]))
            val_acc[0] += np.sum(reading_err < utils.scale_intervals[meter_type])
            val_acc[1] += np.sum(reading_err < 2 * utils.scale_intervals[meter_type])
            pass
        pass
    val_acc /= len(dataloader.dataset)
    return val_acc

def train(dataloaders, epoch = 60, eps = 1e-4):
    train_loader, val_loader = dataloaders
    meter_type = train_loader.dataset.meter_type

    model = ModelOrientResNet50(meter_type)
    model = model.to(device = device, dtype = dtype)
    optimizer = torch.optim.Adam(model.parameters(),1e-4)

    best_model, best_val, best_epoch = None, 0, 0
    for i in range(epoch):
        model.train()
        loss_train = 0
        train_acc = np.zeros(2)
        for sample in train_loader:
            cossins = model(sample["im"])
            inner_prod = torch.sum(cossins * sample["cossin"], dim = 1)
            angle_err = torch.acos(torch.clamp(inner_prod,-1 + eps,1 - eps))
            loss = torch.mean(angle_err)
            loss_train += loss.item()
            reading_err = (angle_err.detach().cpu().numpy() *
                           (utils.reading_ranges[meter_type,1] - utils.reading_ranges[meter_type,0]) /
                           (utils.theta_ranges[meter_type,1] - utils.theta_ranges[meter_type,0]))
            train_acc[0] += np.sum(reading_err < utils.scale_intervals[meter_type])
            train_acc[1] += np.sum(reading_err < 2 * utils.scale_intervals[meter_type])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass
        loss_train /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        
        val_acc = evaluate(model,val_loader)
        if val_acc[0] > best_val:
            best_val = val_acc[0]
            best_model = copy.deepcopy(model)
            best_epoch = i
            pass
        with open(log_file,'a') as f:
            # At the beginning, the training loss (angle error) should be near to $E(|X - Y|) = (\thetta_\max - \thetta_\min) / 3, X,Y \tilde U[\thetta_\min,\thetta_\max]$, 
            # and the training accuracy should be $P(|X - Y| < \frac{r_0}{r_{\max} - r_{\min}}) = 1 - (1 - \frac{r_0}{r_{\max} - r_{\min}})^2$, 
            # where $r_0$ is the reading interval, and $r_\max, r_\min$ the maximal and the minimal reading values.
            # In the training period, the training accuracy can be estimated from the training loss (angle error) by
            # $P(|Z| < \frac{r_0(\theta_\max - \theta_\min)}{r_{\max} - r_{\min}}) = \frac{r_0(\theta_\max - \theta_\min)}{2l(r_{\max} - r_{\min})}, Z \tilds U[0, 2l], E(|Z|) = l$ (which is the loss)
            f.write("\t{:3d}: train_loss = {:.5f}, "
                    "train_acc1 = {:.3f}, train_acc2 = {:.3f}, "
                    "val_acc1 = {:.3f}, val_acc2 = {:.3f}\n".format(i,loss_train,train_acc[0],train_acc[1],val_acc[0],val_acc[1]))
            pass
        pass
    torch.save(best_model.state_dict(),"models/{}_{}_{}_v5.pth".format(os.path.basename(__file__).split('.')[0], "ABCDE"[meter_type], best_epoch))
    return best_model

def test(model, test_loader):
    test_acc = evaluate(model,test_loader)
    with open(log_file,'a') as f:
        f.write("\ttest_acc1 = {:.3f}, test_acc2 = {:.3f}\n".format(test_acc[0],test_acc[1]))
        pass
    return

def train_meter_reader(meter_type, trial):
    if trial:
        # attempt running if true, to see if there are any errors or code bugs
        epoch = 20
    else:
        # run through the whole dataset when nothing is going wrong
        epoch = 120
    train_loader, val_loader, test_loader = get_dataloaders(meter_type, trial = trial)
    model = train((train_loader, val_loader), epoch)
    test(model, test_loader)
    return

def main():
    for meter_type in range(5):
        with open(log_file,'a') as f:
            f.write("{}\n".format("ABCDE"[meter_type]))
            pass    
        train_meter_reader(meter_type, trial = False)
        pass
    return

if __name__ == "__main__":
    main()
