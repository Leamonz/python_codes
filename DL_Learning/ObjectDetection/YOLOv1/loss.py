import torch
import torch.nn as nn
from utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20, lambda_coord=0.5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = 0.5
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        ious = []
        for i in range(self.B):
            iou = intersection_over_union(predictions[..., (self.C + self.B * i + 1):(self.C + self.B * i + 5)],
                                          targets[..., (self.C + 1):(self.C + 5)])
            ious.append(iou.unsqueeze(0))
        ious = torch.cat(ious, dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        # object_i
        exists_box = targets[..., 20].unsqueeze(3)
        # BOX COORDINATES LOSS
        box_predictions = exists_box * predictions[...,
                                       (self.C + self.B * best_box + 1):(self.C + self.B * best_box + 5)]

        box_targets = exists_box * predictions[..., (self.C + 1):(self.C + 5)]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))

        # OBJECT LOSS
        # (N, S, S, 1) -> (N*S*S)
        pred_box = predictions[..., (self.C + self.B * best_box + 1):(self.C + self.B * best_box + 5)]
        # (N, S, S, 1) -> (N*S*S, 1)
        object_loss = self.mse(torch.flatten(exists_box * pred_box),
                               torch.flatten(exists_box * targets[..., self.C:(self.C + 1)]))

        # NO OBJECT LOSS
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = 0
        for i in range(self.B):
            no_object_loss += self.mse(torch.flatten(
                (1 - exists_box) * predictions[..., (self.C + self.B * i + 1):(self.C + self.B * i + 5)]),
                torch.flatten((1 - exists_box) * targets[..., self.C:(self.C + 1)]),
                start_dim=1)

        # CLASS LOSS
        # (N, S, S, C) -> (N*S*S, C)
        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
                              torch.flatten(exists_box * targets[..., :self.C], end_dim=-2))

        loss = self.lambda_coord * box_loss \
               + object_loss \
               + self.lambda_noobj * no_object_loss \
               + class_loss

        return loss
