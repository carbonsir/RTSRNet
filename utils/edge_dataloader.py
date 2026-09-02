
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def _list_files(root):
    return sorted([f for f in os.listdir(root) if f.lower().endswith(_EXTS)])

def _match(root, name):
    stem = os.path.splitext(name)[0]
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        p = os.path.join(root, stem + ext)
        if os.path.exists(p):
            return p
    return None

class TrainEdgeDataset(Dataset):
    def __init__(self, image_root, gt_root, dep_root, edge_root, train_size=384,
                 rVFlip=True, rCrop=True, rRotate=True, colorEnhance=True):
        self.image_root, self.gt_root, self.dep_root, self.edge_root = image_root, gt_root, dep_root, edge_root
        self.train_size = train_size
        self.rVFlip, self.rCrop, self.rRotate, self.colorEnhance = rVFlip, rCrop, rRotate, colorEnhance
        self.names = _list_files(image_root)
        self.rgb_t = transforms.Compose([
            transforms.Resize((train_size, train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.gray_t = transforms.Compose([transforms.Resize((train_size, train_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def _open(self, path, mode):
        return Image.open(path).convert(mode)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = self._open(os.path.join(self.image_root, name), 'RGB')
        gt = self._open(_match(self.gt_root, name), 'L')
        dep = self._open(_match(self.dep_root, name), 'L')
        edge_p = _match(self.edge_root, name)
        edge = self._open(edge_p, 'L') if edge_p is not None else gt.copy()

        if self.rVFlip and random.random() < 0.5:
            img = TF.hflip(img); gt = TF.hflip(gt); dep = TF.hflip(dep); edge = TF.hflip(edge)
        if self.rVFlip and random.random() < 0.1:
            img = TF.vflip(img); gt = TF.vflip(gt); dep = TF.vflip(dep); edge = TF.vflip(edge)
        if self.rCrop and random.random() < 0.5:
            w, h = img.size
            cw = random.randint(int(0.85*w), w)
            ch = random.randint(int(0.85*h), h)
            if cw < w and ch < h:
                x1 = random.randint(0, w-cw); y1 = random.randint(0, h-ch)
                img = TF.crop(img, y1, x1, ch, cw)
                gt = TF.crop(gt, y1, x1, ch, cw)
                dep = TF.crop(dep, y1, x1, ch, cw)
                edge = TF.crop(edge, y1, x1, ch, cw)
        if self.rRotate and random.random() < 0.3:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle); gt = TF.rotate(gt, angle); dep = TF.rotate(dep, angle); edge = TF.rotate(edge, angle)
        if self.colorEnhance and random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8,1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8,1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8,1.2))
            img = TF.adjust_sharpness(img, random.uniform(0.8,1.5))

        img = self.rgb_t(img)
        gt = (self.gray_t(gt) > 0.5).float()
        dep = self.gray_t(dep)
        edge = (self.gray_t(edge) > 0.5).float()
        return idx, img, gt, dep, edge

class TestEdgeDataset(Dataset):
    def __init__(self, image_root, gt_root, dep_root, edge_root=None, test_size=384):
        self.image_root, self.gt_root, self.dep_root, self.edge_root = image_root, gt_root, dep_root, edge_root
        self.names = _list_files(image_root)
        self.rgb_t = transforms.Compose([
            transforms.Resize((test_size, test_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.gray_t = transforms.Compose([transforms.Resize((test_size, test_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def _open(self, path, mode):
        return Image.open(path).convert(mode)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = self._open(os.path.join(self.image_root, name), 'RGB')
        gt = self._open(_match(self.gt_root, name), 'L')
        dep = self._open(_match(self.dep_root, name), 'L')
        edge_p = _match(self.edge_root, name) if self.edge_root is not None and os.path.isdir(self.edge_root) else None
        edge = self._open(edge_p, 'L') if edge_p is not None else gt.copy()

        img_t = self.rgb_t(img)
        dep_t = self.gray_t(dep)
        gt_t = self.gray_t(gt)
        edge_t = self.gray_t(edge)
        return idx, img_t, gt_t, dep_t, edge_t, gt_t.numpy().squeeze(), name
