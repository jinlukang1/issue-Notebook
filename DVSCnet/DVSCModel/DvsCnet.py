import torch
import torch.nn as nn

# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class DvsCnet(nn.Module):
	def __init__(self, num_classes=1000):
		super(DvsCnet, self).__init__()

		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)#stride默认为1
		self.relu1_1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.relu1_2 = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.relu2_1 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.relu2_2 = nn.ReLU(inplace=True)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.relu3_1 = nn.ReLU(inplace=True)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu3_2 = nn.ReLU(inplace=True)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu3_3 = nn.ReLU(inplace=True)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.relu4_1 = nn.ReLU(inplace=True)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu4_2 = nn.ReLU(inplace=True)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu4_3 = nn.ReLU(inplace=True)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu5_1 = nn.ReLU(inplace=True)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu5_2 = nn.ReLU(inplace=True)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu5_3 = nn.ReLU(inplace=True)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.features = nn.Sequential(self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2, self.pool1,
									self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2, self.pool2,
									self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2, self.conv3_3, self.relu3_3, self.pool3, 
									self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2, self.conv4_3, self.relu4_3, self.pool4, 
									self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5
									)

		self.line1 = nn.Linear(512 * 7 * 7, 4096)
		self.line2 = nn.Linear(4096, 4096)
		self.line3 = nn.Linear(4096, num_classes)
		self.classifier = nn.Sequential(
			self.line1,
			nn.ReLU(True),
			nn.Dropout(),
			self.line2,
			nn.ReLU(True),
			nn.Dropout()
		)

		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		x = self.line3(x)
		return x



