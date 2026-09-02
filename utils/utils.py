import math


def normalize(img):
   """
   img: torch.tensor
   normalize img to [0, 1]
   """
   img = (img - img.min()) / (img.max() - img.min() + 1e-8)
   return img

def get_model_complexity(model, inputs, round=3):
    """
    Modified according to https://github.com/GewelsJI/SINet-V2/blob/main/utils/utils.py
    """
    from thop import profile, clever_format
    flops, params = profile(model, inputs=inputs)
    if round is not None:
        flops, params = clever_format([flops, params], f"%.{round}f")
        return flops, params
    return int(flops), int(params)
	
class CosineDecay:
	def __init__(self,
	             optimizer,
	             max_lr,
	             min_lr,
	             max_epoch,
	             test_mode=False):
		self.optimizer = optimizer
		self.max_lr = max_lr
		self.min_lr = min_lr
		self.max_epoch = max(1, int(max_epoch))
		self.test_mode = test_mode

		self.current_lr = max_lr
		self.cnt = 0
		self.scale = (max_lr - min_lr) / 2
		self.shift = (max_lr + min_lr) / 2
		if self.max_epoch > 1:
			self.alpha = math.pi / (self.max_epoch - 1)
		else:
			self.alpha = 0.0

	def step(self):
		self.cnt += 1
		self.current_lr = self.scale * math.cos(self.alpha * self.cnt) + self.shift

		if not self.test_mode:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = self.current_lr

	def get_lr(self):
		return self.current_lr
