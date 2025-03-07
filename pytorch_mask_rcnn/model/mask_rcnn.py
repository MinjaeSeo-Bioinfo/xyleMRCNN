from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc
from .SENet import SEResBackbone
from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer

class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 #@@@@@@@@@@@@ rpn_nms_thresh 0.7 to 0.6
                 rpn_nms_thresh=0.6,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 # 원래 (10, 10, 5, 5)에서 증가
                 box_reg_weights=(15., 15., 7.5, 7.5),
                 #@@@@@@@@@@@@ box_nms_thresh 0.6 to 0.5 , num_detections 200 to 100
                 box_score_thresh=0.1, box_nms_thresh=0.5, box_num_detections=200,
                 # 경계 가중치 매개변수 추가
                 boundary_weight=2.0):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        anchor_sizes = (128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        #------------ RoIHeads --------------------------
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        # boundary_weight 매개변수 전달
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections,
             boundary_weight=boundary_weight)  # 경계 가중치 추가
        
        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        #@@@@@@@@@ self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        self.head.mask_predictor = EnhancedMaskRCNNPredictor(
            out_channels, layers, dim_reduced, num_classes
            )
        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
        
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        
        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result
        
        
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta        
                
#@@@@@@@@@ add skip connection
class EnhancedMaskRCNNPredictor(nn.Module):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        super().__init__()
        
        # 초기 컨볼루션 레이어들
        self.initial_convs = nn.ModuleList()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            conv = nn.Sequential(
                nn.Conv2d(next_feature, layer_features, 3, 1, 1),
                nn.BatchNorm2d(layer_features),
                nn.ReLU(inplace=True)
            )
            self.initial_convs.append(conv)
            next_feature = layer_features
        
        # Skip connection을 위한 1x1 컨볼루션
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(layer_features, dim_reduced, 1) for layer_features in layers
        ])
        
        # 업샘플링 및 특징 통합
        self.upsample_conv = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        self.final_relu = nn.ReLU(inplace=True)
        
        # 최종 분류 레이어
        self.mask_logits = nn.Conv2d(dim_reduced * 2, num_classes, 1, 1, 0)
    
    def forward(self, x):
        # 중간 특징 저장을 위한 리스트
        intermediate_features = []
        
        # 각 레이어 통과 및 특징 추출
        for conv, skip_conv in zip(self.initial_convs, self.skip_convs):
            x = conv(x)
            intermediate_feature = skip_conv(x)
            intermediate_features.append(intermediate_feature)
        
        # 최종 업샘플링
        x = self.upsample_conv(x)
        x = self.final_relu(x)
        
        # Skip connection 통합
        # 마지막 중간 특징 크기 조정 (필요한 경우)
        skip_feature = intermediate_features[-1]
        
        # 크기가 맞지 않을 경우 보간으로 크기 맞추기
        if skip_feature.shape[2:] != x.shape[2:]:
            skip_feature = F.interpolate(skip_feature, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 특징 맵 연결
        x = torch.cat([x, skip_feature], dim=1)
        
        # 최종 마스크 로짓 생성
        mask_logits = self.mask_logits(x)
        
        return mask_logits


def maskrcnn_se_resnet50(pretrained, num_classes, pretrained_backbone=True, boundary_weight=2.0):
    """
    SE 블록과 Depthwise Separable Convolution을 활용한 ResNet-50 백본을 갖는 Mask R-CNN 모델을 생성합니다.
    
    Arguments:
        pretrained (bool): COCO에서 사전 훈련된 가중치를 사용할지 여부.
        num_classes (int): 클래스 수(배경 포함).
        pretrained_backbone (bool): 백본에 사전 훈련된 가중치를 사용할지 여부.
        boundary_weight (float): 마스크 손실에서 경계 픽셀의 가중치.
    """
    # SE-ResNet 백본 생성
    backbone = SEResBackbone('resnet50', pretrained=False)
    model = MaskRCNN(backbone, num_classes, boundary_weight=boundary_weight)
    
    if pretrained:
        print("COCO 사전 훈련 가중치 로드 중...")
        # COCO 사전 훈련 가중치 로드
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        
        # COCO 가중치 다운로드
        coco_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        
        # 현재 모델의 state_dict
        model_state_dict = model.state_dict()
        
        # 백본의 ResNet 부분에만 가중치 로드하기 위한 사전 작업
        pretrained_backbone_dict = {}
        se_layers = []
        
        # 백본의 ResNet 부분과 SE 블록 부분 구분
        for name, _ in model_state_dict.items():
            # SE 블록 및 Depthwise 레이어는 제외
            if 'se_block' in name or 'depthwise' in name or 'pointwise' in name:
                se_layers.append(name)
        
        # COCO 가중치와 현재 모델 state_dict 매핑
        for name, param in model_state_dict.items():
            # 백본의 원래 ResNet 부분에만 COCO 가중치 적용
            if name in coco_state_dict and name not in se_layers:
                # 크기가 맞는 경우에만 가중치 복사
                if param.shape == coco_state_dict[name].shape:
                    pretrained_backbone_dict[name] = coco_state_dict[name]
                    print(f"로드됨: {name}")
                else:
                    print(f"크기 불일치로 건너뜀: {name}")
        
        # SE 블록 및 Depthwise 부분 외 나머지 가중치 로드
        for name, param in coco_state_dict.items():
            # 백본 부분이 아니면서 현재 모델에 존재하는 레이어
            if 'backbone' not in name and name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    pretrained_backbone_dict[name] = param
                    print(f"로드됨(백본 외): {name}")
                else:
                    print(f"크기 불일치로 건너뜀(백본 외): {name}")
        
        # strict=False로 설정하여 SE 블록 및 불일치하는 레이어는 무시
        model.load_state_dict(pretrained_backbone_dict, strict=False)
        print(f"총 {len(pretrained_backbone_dict)}/{len(model_state_dict)} 레이어의 가중치가 로드되었습니다.")
    
    return model

