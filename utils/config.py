import os
import torch


class Config:
    def __init__(self):
        self.dp = DataPath()
        self.num_workers = 8
        self.trainsize = 384
        self.model_name = 'RTSRNet'
        self.result_path = './results'
        self.dir_name = 'RTSRNet'
        self.save_dir = os.path.join(self.result_path, self.dir_name)
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        # Training defaults retained from the provided v4b code path.
        self.epochs = 180
        self.batch_size = 16
        self.weight_decay = 1e-4
        self.learning_rate = 3e-4
        self.min_lr = 1e-6
        self.backbone_lr_mult = 0.2
        self.save_interval = 5
        self.ema_decay = 0.999


class DataPath:
    def __init__(self):
        self.dataset_dir = '/root/autodl-tmp/data'
        self.refresh()

    def refresh(self):
        self.train_imgs = os.path.join(self.dataset_dir, 'TrainDataset', 'Imgs')
        self.train_masks = os.path.join(self.dataset_dir, 'TrainDataset', 'GT')
        self.train_depth = os.path.join(self.dataset_dir, 'TrainDataset', 'Depth')
        self.train_edges = os.path.join(self.dataset_dir, 'TrainDataset', 'Edge')
        for ds in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']:
            setattr(self, f'test_{ds}_imgs', os.path.join(self.dataset_dir, 'TestDataset', ds, 'Imgs'))
            setattr(self, f'test_{ds}_masks', os.path.join(self.dataset_dir, 'TestDataset', ds, 'GT'))
            setattr(self, f'test_{ds}_depth', os.path.join(self.dataset_dir, 'TestDataset', ds, 'Depth'))
            setattr(self, f'test_{ds}_edges', os.path.join(self.dataset_dir, 'TestDataset', ds, 'Edge'))
