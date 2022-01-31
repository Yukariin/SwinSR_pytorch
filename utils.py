from PIL import Image
import copy
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def uint2single(img):
    return np.float32(img/255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    rlt = np.clip(rlt, 0, 255)
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


class ImageSplitter:
    def __init__(self, seg_size=48, scale=2, pad_size=3):
        self.seg_size = seg_size
        self.scale = scale
        self.pad_size = pad_size
        self.channel = 0
        self.height = 0
        self.width = 0

    def split(self, img):
        if isinstance(img, np.ndarray) or isinstance(img, Image.Image):
            img_tensor = TF.to_tensor(img).unsqueeze(0)
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            raise ValueError('This parameter must be a ndarray/Image/Tensor')

        _, c, h, w = img_tensor.size()
        self.channel = c
        self.height = h
        self.width = w

        pad_h = (h // self.seg_size + 1) * self.seg_size - h
        pad_w = (w // self.seg_size + 1) * self.seg_size - w

         # make sure the image is divisible into regular patches
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')

        # add padding around the image to simplify computations
        img_tensor = F.pad(img_tensor, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 'reflect')

        _, _, h, w = img_tensor.size()
        self.height_padded = h
        self.width_padded = w

        patches = []
        for i in range(self.pad_size, h-self.pad_size, self.seg_size):
            for j in range(self.pad_size, w-self.pad_size, self.seg_size):
                patch = img_tensor[:, :,
                    (i-self.pad_size):min(i+self.pad_size+self.seg_size, h),
                    (j-self.pad_size):min(j+self.pad_size+self.seg_size, w)]
                patches.append(patch)

        return patches

    def merge(self, patches, pil_image=True):
        pad_size = self.scale * self.pad_size
        seg_size = self.scale * self.seg_size
        height = self.scale * self.height
        width = self.scale * self.width
        height_padded = self.scale * self.height_padded
        width_padded = self.scale * self.width_padded

        out = torch.zeros((1, self.channel, height_padded, width_padded))
        patch_tensors = copy.copy(patches)

        for i in range(pad_size, height_padded-pad_size, seg_size):
            for j in range(pad_size, width_padded-pad_size, seg_size):
                patch = patch_tensors.pop(0)
                patch = patch[:, :, pad_size:-pad_size, pad_size:-pad_size]

                _, _, h, w = patch.size()
                out[:, :, i:i+h, j:j+w] = patch

        out = out[:, :, pad_size:height+pad_size, pad_size:width+pad_size]

        if pil_image:
            return TF.to_pil_image(out.clamp(0,1).squeeze(0))
        else:
            return out
