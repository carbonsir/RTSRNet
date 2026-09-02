from torch import nn
from torchvision import models


class EfficientNet_B0(nn.Module):
	def __init__(self, pretrained=True):
		super(EfficientNet_B0, self).__init__()

		model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT) if pretrained else models.efficientnet_b0()

		self.layer1 = model.features[0]
		# main layers: layer2 ~ layer8
		self.layer2 = model.features[1]
		self.layer3 = model.features[2]
		self.layer4 = model.features[3]
		self.layer5 = model.features[4]
		self.layer6 = model.features[5]
		self.layer7 = model.features[6]
		self.layer8 = model.features[7]
		# last conv 1×1
		# self.layer9 = model.features[8]

	def forward(self, x):
		# torch.Size([1, 32, 192, 192])
		out1 = self.layer1(x)
		# torch.Size([1, 16, 192, 192])
		out2 = self.layer2(out1)

		# torch.Size([1, 24, 96, 96])
		out3 = self.layer3(out2)

		# torch.Size([1, 40, 48, 48])
		out4 = self.layer4(out3)

		# torch.Size([1, 80, 24, 24])
		out5 = self.layer5(out4)
		# torch.Size([1, 112, 24, 24])
		out6 = self.layer6(out5)

		# torch.Size([1, 192, 12, 12])
		out7 = self.layer7(out6)
		# torch.Size([1, 320, 12, 12])
		out8 = self.layer8(out7)
		# torch.Size([1, 1280, 12, 12])
		# out9 = self.layer9(out8)

		return out2, out3, out4, out6, out8

	@staticmethod
	def get_stage_channels():
		return [16, 24, 40, 112, 320]
