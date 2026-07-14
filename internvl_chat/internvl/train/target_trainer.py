from transformers import Trainer
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import PreTrainedModel
from typing import Optional, Dict, Union, List


class TargetSFTTrainer(Trainer):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_i = 0

    def compute_y_sample_weights(self, target_y, center_y=112.0, base_weight=1.0, scale=0.15, max_weight=5.0):
        """
        target_y: [B], pixel-space y values
        """
        distance_from_center = torch.abs(target_y.float() - center_y)

        weights = base_weight + scale * distance_from_center
        weights = torch.clamp(weights, max=max_weight)

        return weights

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        inputs["labels"] = None

        target_xy = inputs["raw_labels"]
        target_xy_norm = target_xy / (224 - 1) * 2.0 - 1.0

        outputs = model(inputs)

        # Per-coordinate SmoothL1 loss
        loss_per_coord = F.smooth_l1_loss(
            outputs,
            target_xy_norm,
            reduction="none",
        )  # [B, 2]

        loss_x = loss_per_coord[:, 0].mean()
        loss_y = loss_per_coord[:, 1].mean()

        y_sample_weights = self.compute_y_sample_weights(
            target_y=target_xy[:, 1],
            center_y=112.0,
            base_weight=1.0,
            scale=0.2,
            max_weight=6.0,
        )

        loss_norm = loss_x + y_sample_weights * loss_y

        pred_xy_px = (outputs + 1.0) * 0.5 * (224 - 1)
        loss_px = F.smooth_l1_loss(pred_xy_px, target_xy, reduction="none")
        loss_x_px = loss_px[:, 0].mean()
        loss_y_px = loss_px[:, 1].mean()

        loss_px = loss_x_px + y_sample_weights * loss_y_px

        loss = loss_norm + 0.01 * loss_px

        # target_x = target_xy[:, 0].long().clamp(0, 223)
        # target_y = target_xy[:, 1].long().clamp(0, 223)

        # outputs = model(inputs)

        # x_logits = outputs[:, :, 0].squeeze(dim=-1)
        # y_logits = outputs[:, :, 1].squeeze(dim=-1)

        # loss_x = F.cross_entropy(
        #     x_logits,
        #     target_x,
        # )

        # loss_y = F.cross_entropy(
        #     y_logits,
        #     target_y
        # )

        # # Give y more importance if it is under-learning
        # y_loss_weight = 2.0
        # loss = loss_x + y_loss_weight * loss_y

        if self.batch_i % 100 == 0:
            with torch.no_grad():
                pred_xy_pixel = (outputs.detach() + 1.0) * 0.5 * (224 - 1)

                abs_error = torch.abs(pred_xy_pixel - target_xy)
                err_x = abs_error[:, 0].mean()
                err_y = abs_error[:, 1].mean()
                err_l2 = torch.norm(pred_xy_pixel - target_xy, dim=-1).mean()

                log_msg = (
                    f"batch={self.batch_i} "
                    f"loss={loss.item():.6f} "
                    f"err_x={err_x.item():.3f}px "
                    f"err_y={err_y.item():.3f}px "
                    f"err_l2={err_l2.item():.3f}px\n"
                    f"pred_xy={pred_xy_pixel[:5].detach().cpu().tolist()}\n"
                    f"target_xy={target_xy[:5].detach().cpu().tolist()}\n"
                    "--------------------------------------------------\n"
                )

                print(log_msg)

                with open("training_log.txt", "a") as f:
                    f.write(log_msg)
                    f.flush()

        self.batch_i += 1

        return loss