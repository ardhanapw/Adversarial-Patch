import torch
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow
import kornia as K
import numpy as np
import cv2

from torchvision import transforms
from torch import autograd, optim
import torch.nn.functional as F

from contextlib import nullcontext
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm

from easydict import EasyDict
from patch import PatchTransformer, PatchApplier
from loss import MaxProbExtractor, SaliencyLoss, NPSLoss, TotalVariationLoss
from parser import load_config_object, get_argparser

import time

#Convert YOLO to PyTorch Dataloader
#Compare the output between this and Shrestha et al YOLODataset first
#try to make this function closer to Shrestha et al YOLODataset outputs
class RoboflowDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform or transforms.ToTensor()
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Read corresponding YOLO .txt label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    w_img, h_img = image.size

                    # Convert YOLO format (normalized cx, cy, w, h) to pixel coordinates
                    x_c, y_c = x_center * w_img, y_center * h_img
                    box_w, box_h = width * w_img, height * h_img
                    x_min = x_c - box_w / 2
                    y_min = y_c - box_h / 2
                    x_max = x_c + box_w / 2
                    y_max = y_c + box_h / 2

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))

        # To tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # Apply transform to image (convert to tensor, normalization, etc.)
        image = self.transform(image)

        return image, target

class PatchTrainer:
    #dataset
    def __init__(self ,cfg: EasyDict):
        self.cfg = cfg
        self.device = 'cuda:0' if torch.cuda.is_available else 'cpu'

        model = YOLO("yolo11s.pt")
        self.model = model.eval()

        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, self.dev
        ).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)
        self.prob_extractor = MaxProbExtractor(cfg).to(self.dev)
        self.sal_loss = SaliencyLoss().to(self.dev)
        self.nps_loss = NPSLoss(cfg.triplet_printfile, cfg.patch_size).to(self.dev)
        self.tv_loss = TotalVariationLoss().to(self.dev)

        # freeze entire detection model
        for param in self.model.parameters():
            param.requires_grad = False
        
        
        # set log dir
        cfg.log_dir = os.path.join(cfg.log_dir, f'{time.strftime("%Y%m%d-%H%M%S")}_{cfg.patch_name}')
        self.writer = self.init_tensorboard(cfg.log_dir, cfg.tensorboard_port)
        # save config parameters to tensorboard logs
        for cfg_key, cfg_val in cfg.items():
            self.writer.add_text(cfg_key, str(cfg_val))
        

        #dataset augmentation if needed


        #dataloader
        #compare my function with YOLODataset from Shrestha et al first before running the entire code
        base_dir = "/content/All-Day-Vehicle-Recognition-1/train"
        images_dir = os.path.join(base_dir, "images")
        labels_dir = os.path.join(base_dir, "labels")

        dataset = RoboflowDataset(images_dir, labels_dir, transform=transforms.ToTensor())
        self.train_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2
        )

        self.epoch_length = len(self.train_loader) #can be length of the dataset, or customized
    
    def init_tensorboard(self, log_dir: str = None, port: int = 6006):
        """Initialize tensorboard with optional name."""
        tboard = program.TensorBoard()
        tboard.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
        url = tboard.launch()
        print(f"Tensorboard logger started on {url}")

        if log_dir:
            return SummaryWriter(log_dir)
        return SummaryWriter()

    def generate_patch_rand(self):
        #random RGB patch
        patch_width, patch_height = self.cfg.patch_size
        patch = torch.rand((3, patch_width, patch_height))
        return patch

    def generate_patch_from_img(self, path):
        #RGB patch from image
        img = Image.open(path).convert('RGB')
        img = transforms.Resize(self.cfg.patch_size)(img)
        patch = transforms.ToTensor()(img)
        return patch

    def train(self):
        patch_dir = os.path.join(self.cfg.log_dir, "patches")
        os.makedirs(patch_dir, exist_ok=True)

        #threat model
        loss_target = self.cfg.loss_target
        if loss_target == "obj":
            self.cfg.loss_target = lambda obj, cls: obj
        elif loss_target == "cls":
            self.cfg.loss_target = lambda obj, cls: cls
        elif loss_target in {"obj * cls", "obj*cls"}:
            self.cfg.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(f"Loss target {loss_target} not been implemented")
        
        #random patch or from image
        if self.cfg.patch_src == "random":
            adv_patch = self.generate_patch_rand()
        else:
            adv_patch = self.generate_patch_from_img(self.cfg.patch_src)
        adv_patch.requires_grad = True

        optimizer = optim.Adam([adv_patch], lr=self.cfg.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)

        #start_time = time.time()
        for epoch in range(1, self.cfg.n_epochs + 1):
            out_patch_path = os.path.join(patch_dir, f"e_{epoch}.png")
            ep_loss = 0
            min_tv_loss = torch.tensor(self.cfg.min_tv_loss, device=self.dev)
            zero_tensor = torch.tensor([0], device=self.dev)

            for i_batch, (img_batch, lab_batch) in tqdm(
                enumerate(self.train_loader), desc=f"Running train epoch {epoch}", total=self.epoch_length
            ):
                with autograd.set_detect_anomaly(mode=True if self.cfg.debug_mode else False):
                    img_batch = img_batch.to(self.dev, non_blocking=True)
                    lab_batch = lab_batch.to(self.dev, non_blocking=True)
                    adv_patch = adv_patch.to(self.dev, non_blocking=True)
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

                    if self.cfg.debug_mode:
                        img = p_img_batch[
                            0,
                            :,
                            :,
                        ]
                        img = transforms.ToPILImage()(img.detach().cpu())
                        img.save(os.path.join(self.cfg.log_dir, "train_patch_applied_imgs", f"b_{i_batch}.jpg"))

                    with autocast() if self.cfg.use_amp else nullcontext():
                        output = self.model(p_img_batch)[0]
                        max_prob = self.prob_extractor(output)
                        sal = self.sal_loss(adv_patch) if self.cfg.sal_mult != 0 else zero_tensor
                        nps = self.nps_loss(adv_patch) if self.cfg.nps_mult != 0 else zero_tensor
                        tv = self.tv_loss(adv_patch) if self.cfg.tv_mult != 0 else zero_tensor

                    det_loss = torch.mean(max_prob)
                    sal_loss = sal * self.cfg.sal_mult
                    nps_loss = nps * self.cfg.nps_mult
                    tv_loss = torch.max(tv * self.cfg.tv_mult, min_tv_loss)

                    loss = det_loss + sal_loss + nps_loss + tv_loss
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # keep patch in cfg image pixel range
                    pl, ph = self.cfg.patch_pixel_range
                    adv_patch.data.clamp_(pl / 255, ph / 255)

                    if i_batch % self.cfg.tensorboard_batch_log_interval == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar("total_loss", loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/det_loss", det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/sal_loss", sal_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/nps_loss", nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/tv_loss", tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("misc/epoch", epoch, iteration)
                        self.writer.add_scalar("misc/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image("patch", adv_patch, iteration)
                    if i_batch + 1 < len(self.train_loader):
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, sal_loss, nps_loss, tv_loss, loss
            ep_loss = ep_loss / len(self.train_loader)
            scheduler.step(ep_loss)

            # save patch after every patch_save_epoch_freq epochs
            if epoch % self.cfg.patch_save_epoch_freq == 0:
                img = transforms.ToPILImage(self.cfg.patch_img_mode)(adv_patch)
                img.save(out_patch_path)
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, sal_loss, nps_loss, tv_loss, loss

            #validation function
            """
            # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
            all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
            # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
            all_patch_preds = torch.cat(all_patch_preds)
            asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(
                all_labels, all_patch_preds, class_list=self.cfg.class_list, cls_id=cls_id
            )

            print("Validation metrics for images with patches:")
            print(
                f"\tASR@thres={conf_thresh}: asr_s={asr_s:.3f},  asr_m={asr_m:.3f},  asr_l={asr_l:.3f},  asr_a={asr_a:.3f}"
            )
            """

def main():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = load_config_object(args.config)
    trainer = PatchTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
