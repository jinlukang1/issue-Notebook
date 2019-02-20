import torch


def ShowNet(selection):
	if selection == True:
		NetModel = testnet()
		print(NetModel)

# def main():
# 	main_arg_parser = argparse.ArgumentParser(description="some tests")
# 	main_arg_parser.add_argument("--selection", type=bool, default=False,
# 										help="Show the NetModel")
# 	main_arg_parser.add_argument("--dataroot", type=str, default='/gdata/jinlk/jinlukang/example/DVSC/train'
# 										help="The dataset path")
# 	args = main_arg_parser.parse_args()
# 	selection = args.selection
# 	datapath = args.dataroot
	
# 	traindata = DVSC_dataset(datapath)
# 	train_dataloader = torch.utils.data.DataLoader(traindata, batch)
# 	val_dataloader = 

# 	ShowNet(selection)




if __name__ == "__main__":
	main()