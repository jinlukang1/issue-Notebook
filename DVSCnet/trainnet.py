import sys, os

sys.path.append('/ghome/jinlk/lib')
sys.path.append(os.path.join(os.getcwd()))

import argparse
import torch
import ast
from torchvision import models
from DVSCModel.DvsCnet import DvsCnet
from dataset.dataset import DVSC_dataset
from tensorboardX import SummaryWriter




def get_argument():
	main_arg_parser = argparse.ArgumentParser(description="Train DVSC")
	main_arg_parser.add_argument("--selection", type=ast.literal_eval, default=False,#传bool类型只能进去False
										help="Show the NetModel")
	main_arg_parser.add_argument("--dataroot", type=str,
										help="The dataset path")
	main_arg_parser.add_argument("--train_batch_size", type=int, default = 4,
										help="The batch size")
	main_arg_parser.add_argument("--train_shuffle", type=ast.literal_eval, default = True,
										help="shuffle or not in training")
	main_arg_parser.add_argument("--train_num_workers", type=int, default = 4,
										help="num cpu use")
	main_arg_parser.add_argument("--train_epoch", type=int, default = 10,
										help="num of epoch in training")
	main_arg_parser.add_argument("--model_save_path", type=str,
										help="the path to save models")
	main_arg_parser.add_argument("--use_tensorboard", type=ast.literal_eval, default=False,
										help="whether to use tensorboard")
	main_arg_parser.add_argument("--tblog_path", type=str,
										help="where to save the log")
	main_arg_parser.add_argument("--train_load_path", type=str, 
										help="The pretrained model")
	main_arg_parser.add_argument("--train_lr", type=float, default=0.0001,
										help="The learning rate")
	
	return main_arg_parser.parse_args()
	
def mkdirs():
	args = get_argument()
	if args.use_tensorboard and not os.path.exists(args.tblog_path):
		os.makedirs(args.tblog_path)
	if not os.path.exists(args.model_save_path):
		os.makedirs(args.model_save_path)

def train():
	args = get_argument()
	print(args)
	mkdirs()
	#net
	if args.use_tensorboard:
		tblogger = SummaryWriter(args.tblog_path)

	net = DvsCnet(num_classes=2)

	net.load_state_dict(torch.load(args.train_load_path))

	for parma in net.parameters():
		parma.requires_grad = False

	for index, parma in enumerate(net.classifier.parameters()):
		if index == 6:
			parma.requires_grad = True

	net.cuda()
	#data
	train_data = DVSC_dataset(args.dataroot)
	train_dataloader = torch.utils.data.DataLoader(traindata[:int(0.7*len(train_data))], 
													batch_size=args.train_batch_size,
													shuffle=args.train_shuffle,
													num_workers=args.train_num_workers)
	val_dataloader = torch.utils.data.DataLoader(traindata[int(0.7*len(train_data)):], 
													batch_size=args.train_batch_size,
													shuffle=args.train_shuffle,
													num_workers=args.train_num_workers)
	#loss and optimizer
	criterion = nn.CrossEntropyLoss()
	train_lr = args.train_lr

	itr = 0
	itr_max = len(train_dataloader) * train_epoch

	optimizer = torch.optim.Adam(net.classifier.parameters(), lr=train_lr)#lr默认0.001

	for epoch in range(args.train_epoch):
		net.train(True)
		for i, data_batch in enumerate(train_dataloader):
			input_data, input_label = data_batch
			optimizer.zero_grad()
			outputs = net(input_data)
			loss = criterion(outputs, input_label)
			loss.backward()
			optimizer.step()

			if args.use_tensorboard:
				tblogger.add_scalar('loss', loss.item(), itr)

			print('epoch:{}/{} batch:{}/{} iter:{}/{} lr:{} loss:{:05f}'.format(epoch, 
				args.train_epoch, i, len(traindata[:int(0.7*len(train_data))])) // args.train_batch_size, itr, train_lr, loss.item())

			itr += 1

		if args.use_tensorboard:
			tblogger.add_scalar('epoch_loss', loss.item(), epoch)




	# traindata = DVSC_dataset(datapath)
	# # print(traindata[2])




	# ShowNet(selection)


if __name__ == '__main__':
	train()