import torch
from torchvision import transforms as T
import torch.nn.functional as F

import os
from PIL import Image
from torch.utils.data import DataLoader

from easydict import EasyDict
from patch import PatchTransformerNew, PatchApplier
from parser import load_config_object, get_argparser
from adv_patch_utils.common import pad_to_square

from dataset import YOLODataset

import numpy as np
import cv2

import glob

IMG_EXTNS = {".png", ".jpg", ".jpeg"}

class PatchInspector:
    #dataset
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self.device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'  
        
        self.patch_transformer = PatchTransformerNew(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_proj_coef, cfg.y_proj_coef, self.device
        ).to(self.device)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.device)

        #dataset augmentation if needed
        transforms = None
        if cfg.augment_image:
            transforms = T.Compose(
                [
                    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
                    T.ColorJitter(brightness=0.2, hue=0.04, contrast=0.1),
                    T.RandomAdjustSharpness(sharpness_factor=2),
                ]
            )
        
        #dataloader
        self.train_loader = DataLoader(
            YOLODataset(
                image_dir=cfg.image_dir,
                label_dir=cfg.label_dir,
                max_labels=cfg.max_labels,
                model_in_sz=cfg.model_in_sz,
                transform=transforms,
                filter_class_ids=cfg.objective_class_id,
                min_pixel_area=cfg.min_pixel_area,
                shuffle=True,
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
        )

        self.epoch_length = len(self.train_loader) #can be length of the dataset, or customized in the config

    def generate_patch_rand(self):
        #random RGB patch
        patch_height, patch_width = self.cfg.patch_size
        patch = torch.rand((3, patch_height, patch_width))
        return patch

    def generate_patch_from_img(self, path = 'custom_patch.jpg'):
        #RGB patch from image
        img = Image.open(path).convert('RGB')
        img = T.Resize(self.cfg.patch_size)(img)
        patch = T.ToTensor()(img)
        return patch

    """
    def train(self):
        #random patch or from image
        if self.cfg.patch_src == "random":
            print("on random")
            adv_patch_cpu = self.generate_patch_rand()
        else:
            adv_patch_cpu = self.generate_patch_from_img(self.cfg.patch_src)
        adv_patch_cpu.requires_grad = True

        for img_batch, lab_batch in self.train_loader:
            img_batch = img_batch.to(self.device, non_blocking=True)
            lab_batch = lab_batch.to(self.device, non_blocking=True)
            adv_patch = adv_patch_cpu.to(self.device, non_blocking=True)
            adv_batch_t = self.patch_transformer(
                adv_patch,
                lab_batch,
                self.cfg.model_in_sz,
                use_mul_add_gau=self.cfg.use_mul_add_gau,
                do_transforms=self.cfg.transform_patches,
                do_rotate=self.cfg.rotate_patches,
                rand_loc=self.cfg.random_patch_loc,
            )
            p_img_batch = self.patch_applier(img_batch, adv_batch_t)
            p_img_batch = F.interpolate(p_img_batch, (self.cfg.model_in_sz[0], self.cfg.model_in_sz[1]))
            
            #cek gambar yang sudah dikasih patch
            #if self.cfg.debug_mode:
            img = img_batch[
                0,
                :,
                :,
            ]
            img = T.ToPILImage()(img.detach().cpu())
            #img.save(os.path.join(self.cfg.log_dir, "train_patch_applied_imgs", f"original_{i_batch}.jpg"))
            
            patched = p_img_batch[
                0,
                :,
                :,
            ]
            patched = T.ToPILImage()(patched.detach().cpu())
            #patched.save(os.path.join(self.cfg.log_dir, "train_patch_applied_imgs", f"with_patch_{i_batch}.jpg"))
            
            cv2.imshow('Patched', cv2.cvtColor(np.array(patched), cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)  # Waits indefinitely until a key is pressed
            cv2.destroyAllWindows() # Closes all OpenCV 
    """
    def get_bbox_from_xywh(self, label, W, H):
        cx_n, cy_n, w_n, h_n = label[1:]
        
        cx = cx_n * W
        cy = cy_n * H
        bw = w_n  * W
        bh = h_n  * H
        
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
        return x1, y1, x2, y2
            
    def val(self, patchfile: str, conf_thresh: float = 0.0, nms_thresh: float = 0.5) -> None:
        """Calculates the attack success rate according for the patch with respect to different bounding box areas."""
        
        #try to create custom validator instead
        #the image is loaded successfully, but the ASR failed because of raw model outputs
        #try to process with ultralytics final output
        
        # load patch from file
        patch_img = Image.open(patchfile).convert('RGB')
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        adv_patch_cpu = T.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.device)

        img_paths = glob.glob(os.path.join(self.cfg.image_dir, "*"))
        img_paths = sorted([p for p in img_paths if os.path.splitext(p)[-1] in {".png", ".jpg", ".jpeg"}])
        
        label_paths = glob.glob(os.path.join(self.cfg.label_dir, "*"))
        label_paths = sorted([p for p in label_paths])

        train_t_size_frac = self.patch_transformer.t_size_frac
        self.patch_transformer.t_size_frac = [0.3, 0.3]
        
        target_idx = 1 #1 is car index in my dataset
        
        #don't worry about multiple images in same timestamp, it is the result of augmentation
        for (imgfile, labelfile) in zip(img_paths, label_paths):
            print(imgfile, labelfile)
            img = Image.open(imgfile).convert("RGB")
            padded_img = pad_to_square(img)
            padded_img = T.Resize(self.cfg.model_in_sz)(padded_img)
            padded_img_tensor = T.ToTensor()(padded_img).unsqueeze(0).to(self.device)
            
            labels = np.loadtxt(labelfile, ndmin=2)
            labels = labels[labels[:, 0] == target_idx] #1 is car index in my dataset
            
            #filter label so that not all vehicle is patched
            #refer to https://colab.research.google.com/drive/13RgkAzW64nvNXG8tz2-N19SYt_QLKx31
            #code here...
            
            #diusahakan penempatan patch di objek yang tidak tertutup objek lainnya
            #objek yang tidak terpotong (entering/exiting camera): y < 0.75
            #w/h ratio > 0.5 and < 
            #code here..
            
            if labels.shape[0] != 0:
                #print(labels)
                label_batch = torch.from_numpy(labels).float().unsqueeze(0).to(self.device)
                
                adv_batch_t = self.patch_transformer(
                    adv_patch,
                    label_batch,
                    self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc,
                )
                
                #define custom patch transformer
                
                p_tensor_batch = self.patch_applier(padded_img_tensor, adv_batch_t)
                
                patched = p_tensor_batch[
                    0,
                    :,
                    :,
                ]
                patched = T.ToPILImage()(patched.detach().cpu())
                patched = np.array(patched)
                #print(patched.shape)
                
                for label in labels: #bbox format is in xywh, not xyxy
                    x1, y1, x2, y2 = self.get_bbox_from_xywh(label, 640, 640)
                    cv2.rectangle(patched, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                self.patch_transformer.t_size_frac = train_t_size_frac
                
                #p_tensor to PIL
                #PIL to numpy
                #show numpy
                cv2.imshow('Patched', cv2.cvtColor(patched, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)  # Waits indefinitely until a key is pressed
                
        cv2.destroyAllWindows() # Closes all OpenCV 

def main():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = load_config_object(args.config)
    trainer = PatchInspector(cfg)
    #trainer.train()
    trainer.val('custom_patch.jpg')

if __name__ == "__main__":
    main()
