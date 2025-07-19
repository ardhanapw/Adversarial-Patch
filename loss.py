"""Loss functions used in patch generation."""

from typing import Tuple

import torch
import torch.nn as nn


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(MaxProbExtractor, self).__init__()
        self.config = config

    def forward(self, output: torch.Tensor):
        """Output must be of the shape [batch, -1, 5 + num_cls]"""
        """YOLOv8 outputs [batch, 4 + num_cls, -1] actually"""
        """https://github.com/ultralytics/ultralytics/issues/828"""
        #print(output.shape, (4 + self.config.n_classes))
        
        #If YOLOv8 and YOLOv11 (shaped [batch, 4 + num_cls, -1])
        if self.config.architecture_type in ('YOLO', 'yolo'):
            assert output.size(1) == (4 + self.config.n_classes)
            output = output.permute(0, 2, 1)
        elif self.config.architecture_type in ('DETR', 'RTDETR', 'detr', 'rtdetr', 'RT-DETR', 'rt-detr'):
            assert output.size(2) == (4 + self.config.n_classes)
        
        #If DETR (shaped [batch, 300, num_cls])
        #print(output.shape)
        #assert output.size(2) == (self.config.n_classes)
        
        class_confs = output[..., 4:]

        #if self.config.objective_class_id is not None:
        #    class_confs = class_confs[..., self.config.objective_class_id]
        if self.config.objective_class_id_model is not None:
            class_confs = class_confs[..., self.config.objective_class_id_model] #try for MS COCO 'car'
        else:
            class_confs, _ = class_confs.max(dim=-1)

        max_conf, _ = class_confs.max(dim=1)
        return max_conf 


class SaliencyLoss(nn.Module):
    """
    Implementation of the colorfulness metric as the saliency loss.

    The smaller the value, the less colorful the image.
    Reference: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    """

    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Float Tensor of shape [C, H, W] where C=3 (R, G, B channels)
        """
        assert adv_patch.shape[0] == 3
        r, g, b = adv_patch
        rg = r - g
        yb = 0.5 * (r + g) - b

        mu_rg, sigma_rg = torch.mean(rg) + 1e-8, torch.std(rg) + 1e-8
        mu_yb, sigma_yb = torch.mean(yb) + 1e-8, torch.std(yb) + 1e-8
        sl = torch.sqrt(sigma_rg**2 + sigma_yb**2) + (0.3 * torch.sqrt(mu_rg**2 + mu_yb**2))
        return sl / torch.numel(adv_patch)


class TotalVariationLoss(nn.Module):
    """TotalVariationLoss: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W]
        """
        # calc diff in patch rows
        tvcomp_r = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0)
        # calc diff in patch columns
        tvcomp_c = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        tv = tvcomp_r + tvcomp_c
        return tv / torch.numel(adv_patch)


class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(triplet_scores_fpath, size), requires_grad=False
        )

    def forward(self, adv_patch):
        # calculate euclidean distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = adv_patch - self.printability_array + 0.000001
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath, "r", encoding="utf-8") as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        p_h, p_w = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack(
                [torch.full((p_h, p_w), r), torch.full((p_h, p_w), g), torch.full((p_h, p_w), b)]
            )
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)