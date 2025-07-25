"""Modules for creating adversarial object patch."""

import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair, _quadruple

class MedianPool2d(nn.Module):
    """
    Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
class PatchTransformerNew(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    """

    def __init__(
        self,
        t_size_frac: Union[float, Tuple[float, float]] = 0.3,
        mul_gau_mean: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mul_gau_std: Union[float, Tuple[float, float]] = 0.1,
        alpha: float = 0.2,  # Weighting factor for X-axis projection
        beta: float = 1.0,   # Weighting factor for Y-axis projection
        dev: torch.device = torch.device("cuda:0"),
    ):
        super(PatchTransformerNew, self).__init__()
        # convert to duplicated lists/tuples to unpack and send to np.random.uniform
        self.t_size_frac = [t_size_frac, t_size_frac] if isinstance(t_size_frac, float) else t_size_frac
        self.m_gau_mean = [mul_gau_mean, mul_gau_mean] if isinstance(mul_gau_mean, float) else mul_gau_mean
        self.m_gau_std = [mul_gau_std, mul_gau_std] if isinstance(mul_gau_std, float) else mul_gau_std
        assert (
            len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2
        ), "Range must have 2 values"
        self.alpha = alpha
        self.beta = beta
        self.dev = dev
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(kernel_size=7, same=True)

        self.tensor = torch.FloatTensor if "cpu" in str(dev) else torch.cuda.FloatTensor

    def forward(
        self, adv_patch, lab_batch, model_in_sz, use_mul_add_gau=True, do_transforms=True, do_rotate=True, rand_loc=True
    ):

        # add gaussian noise to reduce contrast with a stohastic process
        p_c, p_h, p_w = adv_patch.shape
        if use_mul_add_gau:
            mul_gau = torch.normal(
                np.random.uniform(*self.m_gau_mean),
                np.random.uniform(*self.m_gau_std),
                (p_c, p_h, p_w),
                device=self.dev,
            )
            add_gau = torch.normal(0, 0.001, (p_c, p_h, p_w), device=self.dev)
            adv_patch = adv_patch * mul_gau + add_gau
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        m_h, m_w = model_in_sz
        # Determine size of padding
        pad = (m_w - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1
        )  # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        
        # Apply tailored projection to bounding box centers
        x_center = lab_batch[..., 1]  # Normalized X-center [0,1]
        y_center = lab_batch[..., 2]  # Normalized Y-center [0,1]
        
        # x_center < 0.3 adalah kendaraan di jalur kiri
        # jalur kanan koefisiennya dikali/bagi 2 karena lebih ter-distort
        # proyeksi patch untuk penempatan patch secara dinamis
        max_x = torch.max(x_center, 1 - x_center)
        
        x_left = x_center + (self.alpha*2) * (y_center * max_x)**2 * (x_center - 0.3)
        y_left = y_center + self.beta * (y_center**2) * (y_center - 0.5)
        
        x_right = x_center + (self.alpha) * (y_center * max_x)**2 * (x_center - 0.3)
        y_right = y_center + (self.beta/2) * (y_center**2) * (y_center - 0.5)
        
        is_left = x_center < 0.3
        
        x_proj = torch.where(is_left, x_left, x_right)
        y_proj = torch.where(is_left, y_left, y_right)
        
        # Update bounding box centers with projected values
        lab_batch[..., 1] = x_proj
        lab_batch[..., 2] = y_proj
        
        # Contrast, brightness and noise transforms
        if do_transforms:
            # Create random contrast tensor
            contrast = self.tensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random brightness tensor
            brightness = self.tensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random noise tensor
            noise = self.tensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            adv_batch = adv_batch * contrast + brightness + noise

            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = lab_batch[..., 0].unsqueeze(-1)  # equiv to torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, p_c)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = self.tensor(cls_mask.size()).fill_(1)

        # Pad patch and mask to image dimensions
        patch_pad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = patch_pad(adv_batch)
        msk_batch = patch_pad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = lab_batch.size(0) * lab_batch.size(1)
        if do_rotate:
            angle = self.tensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = self.tensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = self.tensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * m_w
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * m_w
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * m_w
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * m_w
        tsize = np.random.uniform(*self.t_size_frac)
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(tsize)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(tsize)) ** 2)
        )

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        
        #print(target_x, target_y)
        target_width  = lab_batch_scaled[:, :, 3] * 0.6 # frac_x
        target_height = lab_batch_scaled[:, :, 4] * 0.15 # frac_y

        scale_x = target_width / current_patch_size
        scale_y = target_height / current_patch_size
        # Theta = rotation/rescale matrix
        # Theta = input batch of affine matrices with shape (N×2×3) for 2D or (N×3×4) for 3D
        theta = self.tensor(anglesize, 2, 3).fill_(0)
        theta[:,0,0] = cos / scale_x
        theta[:,0,1] = sin / scale_x
        theta[:,1,0] = -sin / scale_y
        theta[:,1,1] = cos / scale_y
        theta[:,0,2] = tx * cos / scale_x + ty * sin / scale_x
        theta[:,1,2] = -tx * sin / scale_y + ty * cos / scale_y
        """
        theta = self.tensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
        """

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid, padding_mode='border')
        msk_batch_t = F.grid_sample(msk_batch, grid, mode='nearest', padding_mode='border')

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    """

    def __init__(
        self,
        t_size_frac: Union[float, Tuple[float, float]] = 0.3,
        mul_gau_mean: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mul_gau_std: Union[float, Tuple[float, float]] = 0.1,
        x_off_loc: Tuple[float, float] = [-0.25, 0.25],
        y_off_loc: Tuple[float, float] = [-0.25, 0.25],
        dev: torch.device = torch.device("cuda:0"),
    ):
        super(PatchTransformer, self).__init__()
        # convert to duplicated lists/tuples to unpack and send to np.random.uniform
        self.t_size_frac = [t_size_frac, t_size_frac] if isinstance(t_size_frac, float) else t_size_frac
        self.m_gau_mean = [mul_gau_mean, mul_gau_mean] if isinstance(mul_gau_mean, float) else mul_gau_mean
        self.m_gau_std = [mul_gau_std, mul_gau_std] if isinstance(mul_gau_std, float) else mul_gau_std
        assert (
            len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2
        ), "Range must have 2 values"
        self.x_off_loc = x_off_loc
        self.y_off_loc = y_off_loc
        self.dev = dev
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(kernel_size=7, same=True)

        self.tensor = torch.FloatTensor if "cpu" in str(dev) else torch.cuda.FloatTensor

    def forward(
        self, adv_patch, lab_batch, model_in_sz, use_mul_add_gau=True, do_transforms=True, do_rotate=True, rand_loc=True
    ):

        # add gaussian noise to reduce contrast with a stohastic process -> from shrestha
        # shrestha patch printability cannot be confirmed
        # lanjut metode thys dkk.
        
        p_c, p_h, p_w = adv_patch.shape
        if use_mul_add_gau:
            mul_gau = torch.normal(
                np.random.uniform(*self.m_gau_mean),
                np.random.uniform(*self.m_gau_std),
                (p_c, p_h, p_w),
                device=self.dev,
            )
            add_gau = torch.normal(0, 0.001, (p_c, p_h, p_w), device=self.dev)
            adv_patch = adv_patch * mul_gau + add_gau
        
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        m_h, m_w = model_in_sz
        # Determine size of padding
        pad = (m_w - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1
        )  # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms
        if do_transforms:
            # Create random contrast tensor
            contrast = self.tensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random brightness tensor
            brightness = self.tensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random noise tensor
            noise = self.tensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            adv_batch = adv_batch * contrast + brightness + noise

            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = lab_batch[..., 0].unsqueeze(-1)  # equiv to torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = self.tensor(cls_mask.size()).fill_(1)

        # Pad patch and mask to image dimensions
        patch_pad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = patch_pad(adv_batch)
        msk_batch = patch_pad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = lab_batch.size(0) * lab_batch.size(1)
        if do_rotate:
            angle = self.tensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = self.tensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = self.tensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * m_w
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * m_w
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * m_w
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * m_w
        tsize = np.random.uniform(*self.t_size_frac)
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(tsize)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(tsize)) ** 2)
        )

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if rand_loc:
            off_x = targetoff_x * (self.tensor(targetoff_x.size()).uniform_(*self.x_off_loc))
            target_x = target_x + off_x
            off_y = targetoff_y * (self.tensor(targetoff_y.size()).uniform_(*self.x_off_loc))
            target_y = target_y + off_y
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation/rescale matrix
        # Theta = input batch of affine matrices with shape (N×2×3) for 2D or (N×3×4) for 3D
        theta = self.tensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    The patch (adv_batch) has the same size as the image, just is zero everywhere there isn't a patch.
    If patch_alpha == 1 (default), just overwrite the background image values with the patch values.
    Else, blend the patch with the image
    See: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
         https://stackoverflow.com/questions/49737541/merge-two-images-with-alpha-channel/49738078
        I = \alpha F + (1 - \alpha) B
            F = foregraound (patch, or adv_batch)
            B = background (image, or img_batch)
    """

    def __init__(self, patch_alpha: float = 1):
        super(PatchApplier, self).__init__()
        self.patch_alpha = patch_alpha

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # replace image values with patch values
            if self.patch_alpha == 1:
                img_batch = torch.where((adv == 0), img_batch, adv)
            # alpha blend
            else:
                # get combo of image and adv
                alpha_blend = self.patch_alpha * adv + (1.0 - self.patch_alpha) * img_batch
                # apply alpha blend where the patch is non-zero
                img_batch = torch.where((adv == 0), img_batch, alpha_blend)

        return img_batch