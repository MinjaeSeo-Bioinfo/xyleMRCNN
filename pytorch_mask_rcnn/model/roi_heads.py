import torch
import torch.nn.functional as F
from torch import nn

from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss


def boundary_sensitive_loss(mask_logit, mask_target, boundary_weight=2.0):
    """
    Compute boundary-sensitive loss for mask prediction.
    
    Args:
        mask_logit: The predicted mask logits
        mask_target: The ground truth mask
        boundary_weight: Weight to apply to boundary pixels
    
    Returns:
        Weighted binary cross entropy loss
    """
    # Create boundary mask
    # Dilate and erode to find boundaries
    kernel_size = 3
    mask_float = mask_target.float()
    
    # Use max_pool2d for dilation
    dilated = torch.nn.functional.max_pool2d(
        mask_float, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=kernel_size//2
    )
    # Use max_pool2d on inverted mask for erosion
    eroded = 1.0 - torch.nn.functional.max_pool2d(
        1.0 - mask_float,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size//2
    )
    # Boundary is the difference between dilated and eroded
    boundary_mask = (dilated - eroded).clamp(0, 1)
    
    # Create weight map giving higher weight to boundary pixels
    weights = torch.ones_like(mask_target, dtype=torch.float)
    weights = weights + boundary_mask * (boundary_weight - 1)
    
    # Apply weighted binary cross entropy
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        mask_logit, 
        mask_target.float(), 
        weight=weights,
        reduction='mean'
    )
    return loss

# modify the existing maskrcnn_loss function
def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask, boundary_weight=2.0):
    """
    Compute the mask prediction loss with boundary sensitivity.
    """
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    
    # Use boundary-sensitive loss instead of standard BCE
    if boundary_weight > 1.0:
        mask_loss = boundary_sensitive_loss(
            mask_logit[idx, label], 
            mask_target,
            boundary_weight=boundary_weight
        )
    else:
        # Fall back to original loss if boundary weight is not enabled
        mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            mask_logit[idx, label], 
            mask_target
        )
    return mask_loss

# def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
#     matched_idx = matched_idx[:, None].to(proposal)
#     roi = torch.cat((matched_idx, proposal), dim=1)
            
#     M = mask_logit.shape[-1]
#     gt_mask = gt_mask[:, None].to(roi)
#     mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

#     idx = torch.arange(label.shape[0], device=label.device)
#     mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
#     return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections,
                 boundary_weight=2.0):  #@@@ Add boundary_weight parameter
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        self.boundary_weight = boundary_weight
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposal, target):
        gt_box = target['boxes']
        gt_label = target['labels']
        proposal = torch.cat((proposal, gt_box))
        
        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))
        
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0
        
        return proposal, matched_idx, label, regression_target
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)
        
        boxes = []
        labels = []
        scores = []
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)
            
            box, score = process_box(box, score, image_shape, self.min_size)
            
            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
            
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results
    
    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)
        
        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)
        
        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)
            
        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]
                print(f"양성 샘플 수: {num_pos}")  # 디버깅 출력
                
                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]
                
                if mask_proposal.shape[0] == 0:
                    print("경고: 양성 샘플이 없어 마스크 손실이 0입니다!")  # 디버깅 출력
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['boxes']
                
                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)
            
            if self.training:
                gt_mask = target['masks']
                # Pass boundary_weight to the loss function
                mask_loss = maskrcnn_loss(
                    mask_logit, mask_proposal, pos_matched_idx, 
                    mask_label, gt_mask, self.boundary_weight
                )
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]

                mask_prob = mask_logit.sigmoid()
                result.update(dict(masks=mask_prob))
                
        return result, losses
