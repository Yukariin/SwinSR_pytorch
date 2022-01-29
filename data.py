import cv2
from os import listdir
from os.path import join
import io
import random
import sqlite3

import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import RandomCrop, Resize
from torchvision.transforms import functional as TF

import utils as util


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DatasetFromList(data.Dataset):
    def __init__(self, list_path, patch_size=48, scale_factor=2, interpolation=Image.BICUBIC):
        super().__init__()

        self.samples = [x.rstrip('\n') for x in open(list_path) if is_image_file(x.rstrip('\n'))]
        self.cropper = RandomCrop(patch_size * scale_factor)
        self.resizer = Resize(patch_size, interpolation)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        img = Image.open(sample_path).convert('RGB')

        target = self.cropper(img)
        input = target.copy()
        input = self.resizer(input)

        return TF.to_tensor(input), TF.to_tensor(target)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size=48, scale_factor=2, interpolation=None,
                 rotate=True, hflip=True, vflip=False):
        super().__init__()

        self.samples = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.cropper = RandomCrop(patch_size * scale_factor)
        self.resizer = Resize(patch_size, interpolation)
        self.rotate = rotate
        self.hflip = hflip
        self.vflip = vflip
        self.interpolation = interpolation
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        img = Image.open(sample_path).convert('RGB')

        hr = self.cropper(img)

        if self.interpolation is None:
            interpolation = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        else:
            interpolation = self.interpolation
        lr = hr.resize((self.patch_size,self.patch_size), interpolation)

        if self.rotate and np.random.rand() < 0.5:
            rv = np.random.randint(1, 4)
            lr = TF.rotate(lr, 90 * rv)
            hr = TF.rotate(hr, 90 * rv)
        if self.hflip and np.random.rand() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        if self.vflip and np.random.rand() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)

        return TF.to_tensor(lr), TF.to_tensor(hr)


class SQLDataset(data.Dataset):
    def __init__(self, db_file, db_table='images', lr_col='lr_img', hr_col='hr_img', rotate=True, hflip=True, vflip=False):
        super().__init__()

        self.db_file = db_file
        self.db_table = db_table
        self.lr_col = lr_col
        self.hr_col = hr_col
        self.rotate = rotate
        self.hflip = hflip
        self.vflip = vflip
        self.total_images = self.get_num_rows()

    def get_num_rows(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT MAX(ROWID) FROM {self.db_table}')
            db_rows = cursor.fetchone()[0]

        return db_rows

    def __len__(self):
        return self.total_images

    def __getitem__(self, item):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT {self.lr_col}, {self.hr_col} FROM {self.db_table} WHERE ROWID={item+1}')
            lr, hr = cursor.fetchone()

        lr = Image.open(io.BytesIO(lr)).convert('RGB')
        hr = Image.open(io.BytesIO(hr)).convert('RGB')

        if self.rotate and np.random.rand() < 0.5:
            rv = np.random.randint(1, 4)
            lr = TF.rotate(lr, 90 * rv)
            hr = TF.rotate(hr, 90 * rv)
        if self.hflip and np.random.rand() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        if self.vflip and np.random.rand() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)

        return TF.to_tensor(lr), TF.to_tensor(hr)


class DatasetJPEG(data.Dataset):
    def __init__(self, root, n_channels=3, H_size=128, quality_factor=40, is_color=False):
        super().__init__()

        self.n_channels = n_channels
        self.patch_size = H_size

        self.quality_factor = quality_factor
        self.is_color = is_color

        self.paths_H = [join(root, x) for x in sorted(listdir(root)) if is_image_file(x)]

    def __len__(self):
        return len(self.paths_H)

    def __getitem__(self, index):
        # get H image
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        H, W = img_H.shape[:2]
        self.patch_size_plus = self.patch_size + 8

        # randomly crop a large patch
        rnd_h = random.randint(0, max(0, H - self.patch_size_plus))
        rnd_w = random.randint(0, max(0, W - self.patch_size_plus))
        patch_H = img_H[rnd_h:rnd_h + self.patch_size_plus, rnd_w:rnd_w + self.patch_size_plus, ...]

        # augmentation - flip, rotate
        mode = random.randint(0, 7)
        patch_H = util.augment_img(patch_H, mode=mode)

        # HWC to CHW, numpy(uint) to tensor
        img_L = patch_H.copy()

        # set quality factor
        quality_factor = self.quality_factor

        if self.is_color:  # color image
            img_H = img_L.copy()
            img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
            result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 1)
            img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
        else:
            if random.random() > 0.5:
                img_L = util.rgb2ycbcr(img_L)
            else:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2GRAY)
            img_H = img_L.copy()
            result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0)

        # randomly crop a patch
        H, W = img_H.shape[:2]
        if random.random() > 0.5:
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
        else:
            rnd_h = 0
            rnd_w = 0
        img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]


        img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        return img_L, img_H


class DatasetSR(data.Dataset):
    def __init__(self, root, n_channels=3, scale=4, H_size=96):
        super().__init__()

        self.n_channels = n_channels
        self.sf = scale
        self.patch_size = H_size
        self.L_size = self.patch_size // self.sf

        self.paths_H = [join(root, x) for x in sorted(listdir(root)) if is_image_file(x)]

    def __len__(self):
        return len(self.paths_H)

    def __getitem__(self, index):
        # get H image
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # modcrop
        img_H = util.modcrop(img_H, self.sf)

        # get L image
        H, W = img_H.shape[:2]
        img_L = util.imresize_np(img_H, 1 / self.sf, False)

        # get L/H patch pair
        H, W, C = img_L.shape

        # randomly crop the L patch
        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

        # crop corresponding H patch
        rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
        img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

        # augmentation - flip and/or rotate
        mode = random.randint(0, 7)
        img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # L/H pairs, HWC to CHW, numpy to tensor
        img_L, img_H = util.single2tensor3(img_L), util.single2tensor3(img_H)

        return img_L, img_H


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0
