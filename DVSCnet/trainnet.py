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
import torch.nn as nn




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

def val_acc(model, dataloader, criterion):
	model.eval()
	for i, data_batch in enumerate(dataloader):
		val_input, val_label = data_batch
		val_input = Variable(val_input, volatile=True)
		val_label = Variable(val_input.type(t.LongTensor), volatile=True)
		val_input = val_input.cuda()
		val_label = val_label.cuda()

		val_output = model(val_input)

		_. preds = torch.max(val_output, 1)

		loss = criterion(val_output, val_label)

		running_loss += loss.item()
		running_corrects += torch.sum(preds == val_label)

	epoch_loss = running_loss / len(dataloader)
	epoch_acc = running_corrects / len(dataloader)

	return epoch_loss, epoch_acc


def train():
	args = get_argument()
	print(args)
	mkdirs()
	#net
	if args.use_tensorboard:
		tblogger = SummaryWriter(args.tblog_path)

	net = DvsCnet()
	net.line3 = nn.Linear(4096, 2)
	print('loading pretrained model...')
	#model load
	pretrained_dict = torch.load(args.train_load_path)
	net_dict = net.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
	net_dict.update(pretrained_dict)
	net.load_state_dict(net_dict)
	# for i, j in net.named_parameters():
	# 	print(i)
	print('model loading success!')

	for parma in net.parameters():
		parma.requires_grad = False

	for index, parma in enumerate(net.line3.parameters()):
		parma.requires_grad = True

	net.cuda()
	#data
	train_data = DVSC_dataset(args.dataroot, train=True)
	val_data = DVSC_dataset(args.dataroot, train=False)
	# train_data = train_data.cuda()
	# val_data = val_data.cuda()
	train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size,
													shuffle=args.train_shuffle,
													num_workers=args.train_num_workers)
	val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.train_batch_size,
													shuffle=args.train_shuffle,
													num_workers=args.train_num_workers)
	#loss and optimizer
	criterion = nn.CrossEntropyLoss()
	train_lr = args.train_lr

	itr = 0
	best_acc = 0.0
	itr_max = len(train_dataloader) * args.train_epoch

	optimizer = torch.optim.Adam(net.line3.parameters(), lr=train_lr)#lr默认0.001

	print('start training')
	for epoch in range(args.train_epoch):
		net.train(True)
		for i, data_batch in enumerate(train_dataloader):
			train_input, train_label = data_batch
			train_input = train_input.cuda()
			train_label = train_label.cuda()

			optimizer.zero_grad()
			train_outputs = net(train_input)
			loss = criterion(train_outputs, train_label)
			loss.backward()
			optimizer.step()
			print

			if args.use_tensorboard:
				tblogger.add_scalar('loss', loss.item(), itr)

			print('epoch:{}/{} batch:{}/{} iter:{} lr:{} loss:{:05f}'.format(epoch, 
				args.train_epoch, i, len(train_data) // args.train_batch_size, itr, train_lr, loss.item()))

			itr += 1

		epoch_loss, epoch_acc = val_acc(net, val_dataloader, criterion)

		if epoch_acc > best_acc:
			best_acc = epoch_acc
			torch.save(net.state_dict(), "DVSC_{}".format(epoch))
			print("Checkpoints saved!")



		if args.use_tensorboard:
			tblogger.add_scalar('epoch_loss', epoch_loss, epoch)
			tblogger.add_scalar('epoch_acc', epoch_acc, epoch)

		print('epoch:{}''epoch_acc:{}'.format(epoch, epoch_acc))



# sklearn net.eval
	# traindata = DVSC_dataset(datapath)
	# # print(traindata[2])




	# ShowNet(selection)


if __name__ == '__main__':
	train()