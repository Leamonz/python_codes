import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    # boxes_preds shape: (N, 4) where N represents the number of bboxes
    # boxes_labels shape: (N, 4)
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[:, 0:1] - boxes_preds[:, 2:3] / 2
        box1_y1 = boxes_preds[:, 1:2] - boxes_preds[:, 3:4] / 2
        box1_x2 = boxes_preds[:, 0:1] + boxes_preds[:, 2:3] / 2
        box1_y2 = boxes_preds[:, 1:2] + boxes_preds[:, 3:4] / 2
        box2_x1 = boxes_labels[:, 0:1] - boxes_labels[:, 2:3] / 2
        box2_y1 = boxes_labels[:, 1:2] - boxes_labels[:, 3:4] / 2
        box2_x2 = boxes_labels[:, 0:1] + boxes_labels[:, 2:3] / 2
        box2_y2 = boxes_labels[:, 1:2] + boxes_labels[:, 3:4] / 2

    elif box_format == 'corners':
        box1_x1 = boxes_preds[:, 0:1]
        box1_y1 = boxes_preds[:, 1:2]
        box1_x2 = boxes_preds[:, 2:3]
        box1_y2 = boxes_preds[:, 3:4]
        box2_x1 = boxes_labels[:, 0:1]
        box2_y1 = boxes_labels[:, 1:2]
        box2_x2 = boxes_labels[:, 2:3]
        box2_y2 = boxes_labels[:, 3:4]

    calc_x1 = torch.max(box1_x1, box2_x1)
    calc_y1 = torch.max(box1_y1, box2_y1)
    calc_x2 = torch.min(box1_x2, box2_x2)
    calc_y2 = torch.min(box1_y2, box2_y2)
    intersection = (calc_x2 - calc_x1).clamp(0) * (calc_y2 - calc_y1).clamp(0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection
    return intersection / (union + 1e-6)


