import torch
import ultralytics
from torchvision import transforms as T
from torch import autograd, optim
import torch.nn.functional as F

from contextlib import nullcontext
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from easydict import EasyDict
from patch import PatchTransformerNew, PatchApplier
from loss import MaxProbExtractor, SaliencyLoss, NPSLoss, TotalVariationLoss
from parser import load_config_object, get_argparser
from dataset import YOLODataset

import time

import glob

from utils.general import non_max_suppression, xyxy2xywh
from adv_patch_utils.common import pad_to_square
from models.common import DetectMultiBackend

from patch_tester import PatchTester

import numpy as np


#NPS is not used by Shrestha et al, they use gaussian multiplication and addition instead
#Hence NPS multiplication factor is 0 in .json

class PatchTrainer:
    #dataset
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  

        self.model_wrapper = ultralytics.YOLO(cfg.weights_file).to(self.device)
        #model = DetectMultiBackend(cfg.weights_file, device=self.device, dnn=False, data=None, fp16=False)
        self.model = self.model_wrapper.model.eval()
        
        #self.patch_transformer = PatchTransformer(
        #    cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, self.device
        #).to(self.device)
        self.patch_transformer = PatchTransformerNew(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_proj_coef, cfg.y_proj_coef, self.device
        ).to(self.device)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.device)
        self.prob_extractor = MaxProbExtractor(cfg).to(self.device)
        self.sal_loss = SaliencyLoss().to(self.device)
        self.nps_loss = NPSLoss(cfg.triplet_printfile, cfg.patch_size).to(self.device)
        self.tv_loss = TotalVariationLoss().to(self.device)
        
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
                filter_class_ids=cfg.objective_class_id_dataset,
                min_pixel_area=cfg.min_pixel_area,
                shuffle=True,
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
        )

        self.epoch_length = len(self.train_loader) #can be length of the dataset, or customized in the config
    
    def init_tensorboard(self, log_dir, port):
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
        patch_height, patch_width = self.cfg.patch_size
        patch = torch.rand((3, patch_height, patch_width))
        return patch

    def generate_patch_from_img(self, path = 'custom_patch.jpg'):
        #RGB patch from image
        img = Image.open(path).convert('RGB')
        img = T.Resize(self.cfg.patch_size)(img)
        patch = T.ToTensor()(img)
        return patch

    def train(self):
        #patch output path
        patch_dir = os.path.join(self.cfg.log_dir, "patches")
        os.makedirs(patch_dir, exist_ok=True)
        
        if self.cfg.debug_mode:
            for img_dir in ["train_patch_applied_imgs", "val_clean_imgs", "val_patch_applied_imgs"]:
                os.makedirs(os.path.join(self.cfg.log_dir, img_dir), exist_ok=True)
        
        print(self.cfg.patch_src)
        #random patch or from image
        if self.cfg.patch_src == "random":
            print("on random")
            adv_patch_cpu = self.generate_patch_rand()
        else:
            adv_patch_cpu = self.generate_patch_from_img(self.cfg.patch_src)
        adv_patch_cpu.requires_grad = True

        optimizer = optim.Adam([adv_patch_cpu], lr=self.cfg.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)

        start_time = time.time()
        for epoch in range(1, self.cfg.n_epochs + 1):
            out_patch_path = os.path.join(patch_dir, f"e_{epoch}.png")
            ep_loss = 0
            min_tv_loss = torch.tensor(self.cfg.min_tv_loss, device=self.device)
            zero_tensor = torch.tensor([0], device=self.device)

            for i_batch, (img_batch, lab_batch) in tqdm(
                enumerate(self.train_loader), desc=f"Running train epoch {epoch}", total=self.epoch_length
            ):
                with autograd.set_detect_anomaly(mode=True if self.cfg.debug_mode else False):
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
                    if self.cfg.debug_mode:
                        img = img_batch[
                            0,
                            :,
                            :,
                        ]
                        img = T.ToPILImage()(img.detach().cpu())
                        img.save(os.path.join(self.cfg.log_dir, "train_patch_applied_imgs", f"original_{i_batch}.jpg"))
                        
                        patched = p_img_batch[
                            0,
                            :,
                            :,
                        ]
                        patched = T.ToPILImage()(patched.detach().cpu())
                        patched.save(os.path.join(self.cfg.log_dir, "train_patch_applied_imgs", f"with_patch_{i_batch}.jpg"))
                    #print("before prob")
                    with autocast() if self.cfg.use_amp else nullcontext():
                        output = self.model(p_img_batch)[0]
                        max_prob = self.prob_extractor(output)
                        #sal = self.sal_loss(adv_patch) if self.cfg.sal_mult != 0 else zero_tensor
                        nps = self.nps_loss(adv_patch) if self.cfg.nps_mult != 0 else zero_tensor
                        tv = self.tv_loss(adv_patch) if self.cfg.tv_mult != 0 else zero_tensor

                    det_loss = torch.mean(max_prob)
                    #det_loss = output[..., 4].mean() -> this here is only valid on YOLOv5
                    #sal_loss = sal * self.cfg.sal_mult
                    nps_loss = nps * self.cfg.nps_mult
                    tv_loss = torch.max(tv * self.cfg.tv_mult, min_tv_loss)
                    
                    loss = det_loss + tv_loss + nps_loss #+ sal_loss
                    #print(torch.mean(max_prob), det_loss, sal_loss, tv_loss)
                    ep_loss += loss

                    loss.backward()
                    if (i_batch+1) % 16 == 0 or (i_batch+1) == len(self.train_loader): #batch_size mentok di 2, update optimizer setiap 16 iterasi, gradient setara 32 batch
                        #print("Grad norm:", adv_patch.grad.norm().item())
                        #optimizer
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        
                    # keep patch in cfg image pixel range
                    pl, ph = self.cfg.patch_pixel_range
                    adv_patch.data.clamp_(pl / 255, ph / 255)

                    if i_batch % self.cfg.tensorboard_batch_log_interval == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar("total_loss", loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/det_loss", det_loss.detach().cpu().numpy(), iteration)
                        #self.writer.add_scalar("loss/sal_loss", sal_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/nps_loss", nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/tv_loss", tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("misc/epoch", epoch, iteration)
                        self.writer.add_scalar("misc/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image("patch", adv_patch, iteration)
                    if i_batch + 1 < len(self.train_loader):
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, tv_loss, loss, nps_loss#, sal_loss
            ep_loss = ep_loss / len(self.train_loader)
            scheduler.step(ep_loss)

            # save patch after every patch_save_epoch_freq epochs
            if epoch % self.cfg.patch_save_epoch_freq == 0:
                img = T.ToPILImage('RGB')(adv_patch)
                img.save(out_patch_path)
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss#, sal_loss

            #validation function
            #if all([self.cfg.val_image_dir, self.cfg.val_epoch_freq]) and epoch % self.cfg.val_epoch_freq == 0:
            #    with torch.no_grad():
            #        self.val(epoch, out_patch_path)
    
        print(f"Total training time {time.time() - start_time:.2f}s")

    def val(self, epoch: int, patchfile: str, conf_thresh: float = 0.0, nms_thresh: float = 0.5) -> None:
        """Calculates the attack success rate according for the patch with respect to different bounding box areas."""
        
        #try to create custom validator instead
        #the image is loaded successfully, but the ASR failed because of raw model outputs
        #try to process with ultralytics final output
        
        # load patch from file
        patch_img = Image.open(patchfile).convert('RGB')
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        adv_patch_cpu = T.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.device)

        img_paths = glob.glob(os.path.join(self.cfg.val_image_dir, "*"))
        img_paths = sorted([p for p in img_paths if os.path.splitext(p)[-1] in {".png", ".jpg", ".jpeg"}])

        #train_t_size_frac = self.patch_transformer.t_size_frac
        #self.patch_transformer.t_size_frac = [0.3, 0.3]  # use a frac of 0.3 for validation
        # to calc confusion matrixes and attack success rates later
        all_labels = []
        all_patch_preds = []

        m_h, m_w = self.cfg.model_in_sz
        cls_id = self.cfg.objective_class_id_model
        zeros_tensor = torch.zeros([1, 5]).to(self.device)
        #### iterate through all images ####
        for imgfile in tqdm(img_paths, desc=f"Running val epoch {epoch}"):
            img_name = os.path.splitext(imgfile)[0].split("/")[-1]
            img = Image.open(imgfile).convert("RGB")
            padded_img = pad_to_square(img)
            padded_img = T.Resize(self.cfg.model_in_sz)(padded_img)

            #######################################
            # generate labels to use later for patched image
            padded_img_tensor = T.ToTensor()(padded_img).unsqueeze(0).to(self.device)
            pred = self.model_wrapper(padded_img_tensor)[0]
            boxes = pred.boxes.data
            
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            all_labels.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            labels = []
            for box in boxes:
                cls_id_box = box[-1].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                labels.append([cls_id_box, x_center / m_w, y_center / m_h, width / m_w, height / m_h])

            # save img if debug mode
            if self.cfg.debug_mode:
                padded_img_drawn = PatchTester.draw_bbox_on_pil_image(all_labels[-1], padded_img, self.cfg.class_list)
                padded_img_drawn.save(os.path.join(self.cfg.log_dir, "val_clean_imgs", img_name + ".jpg"))

            # use a filler zeros array for no dets
            label = np.asarray(labels) if labels else np.zeros([1, 5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            #######################################
            # Apply proper patches
            img_fake_batch = padded_img_tensor
            lab_fake_batch = label.unsqueeze(0).to(self.device)

            if len(lab_fake_batch[0]) == 1 and torch.equal(lab_fake_batch[0], zeros_tensor):
                # no det, use images without patches
                p_tensor_batch = padded_img_tensor
            else:
                # transform patch and add it to image
                adv_batch_t = self.patch_transformer(
                    adv_patch,
                    lab_fake_batch,
                    self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc,
                )
                p_tensor_batch = self.patch_applier(img_fake_batch, adv_batch_t)

            pred = self.model_wrapper(p_tensor_batch)[0]
            boxes = pred.boxes.data

            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            all_patch_preds.append(boxes.clone())

            # save properly patched img if debug mode
            if self.cfg.debug_mode:
                p_img_pil = T.ToPILImage("RGB")(p_tensor_batch.squeeze(0).cpu())
                p_img_pil_drawn = PatchTester.draw_bbox_on_pil_image(
                    all_patch_preds[-1], p_img_pil, self.cfg.class_list
                )
                p_img_pil_drawn.save(os.path.join(self.cfg.log_dir, "val_patch_applied_imgs", img_name + ".jpg"))

        
        # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
        all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
        # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
        all_patch_preds = torch.cat(all_patch_preds)
        print(all_labels.shape, all_patch_preds.shape)
        
        #Legacy ASR is problematic
        """
        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(
            all_labels, all_patch_preds, class_list=self.cfg.class_list, cls_id=cls_id
        )

        print("Validation metrics for images with patches:")
        print(
            f"\tASR@thres={conf_thresh}: asr_s={asr_s:.3f},  asr_m={asr_m:.3f},  asr_l={asr_l:.3f},  asr_a={asr_a:.3f}"
        )
        """

        #self.writer.add_scalar("val_asr_per_epoch/area_small", asr_s, epoch)
        #self.writer.add_scalar("val_asr_per_epoch/area_medium", asr_m, epoch)
        #self.writer.add_scalar("val_asr_per_epoch/area_large", asr_l, epoch)
        #self.writer.add_scalar("val_asr_per_epoch/area_all", asr_a, epoch)
        del adv_batch_t, padded_img_tensor, p_tensor_batch
        torch.cuda.empty_cache()
        #self.patch_transformer.t_size_frac = train_t_size_frac
    

def main():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = load_config_object(args.config)
    trainer = PatchTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
