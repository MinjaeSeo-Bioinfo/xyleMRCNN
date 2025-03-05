import os
import torch
import pytorch_mask_rcnn as pmr
import argparse
import bisect
import time
import re
import glob
import json
import numpy as np

def load_rejected_annotations():
    """반려된 어노테이션 ID 목록 로드"""
    rejected_ids = []
    if os.path.exists('rejected_annotations.txt'):
        with open('rejected_annotations.txt', 'r') as f:
            rejected_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    return rejected_ids

class XylemDatasetWithRejected(pmr.datasets.XylemDataset):
    """
    반려된 어노테이션을 제외하거나 가중치를 높인 데이터셋
    """
    def __init__(self, data_dir, split, train=False, rejected_ids=None, rejection_mode='exclude', 
                 hard_negative_weight=2.0, misclassified_weight=3.0):
        super().__init__(data_dir, split, train)
        
        # 반려된 어노테이션 ID
        self.rejected_ids = rejected_ids if rejected_ids is not None else []
        self.rejection_mode = rejection_mode  # 'exclude' 또는 'weight'
        
        # 가중치 설정
        self.hard_negative_weight = hard_negative_weight  # 거짓 양성(False Positive) 가중치
        self.misclassified_weight = misclassified_weight  # 오분류 가중치
        
        # 어노테이션 별 가중치 맵 (rejection_mode='weight'인 경우 사용)
        self.annotation_weights = {}
        
        # 거부된 어노테이션 처리
        if rejection_mode == 'exclude':
            # 거부된 어노테이션 ID를 제외하고 데이터셋 필터링
            self._filter_rejected_annotations()
        elif rejection_mode == 'weight':
            # 거부된 어노테이션에 가중치 적용
            self._apply_annotation_weights()
    
    def _filter_rejected_annotations(self):
        """반려된 어노테이션 ID를 제외한 데이터셋 필터링"""
        if not self.rejected_ids:
            return
        
        # 필터링할 이미지 ID 찾기
        filtered_img_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # 이미지에 반려된 어노테이션이 있는지 확인
            has_rejected = any(ann['id'] in self.rejected_ids for ann in anns)
            
            # 반려된 어노테이션이 없는 이미지만 포함
            if not has_rejected:
                filtered_img_ids.append(img_id)
        
        # 필터링된 이미지 ID로 데이터셋 업데이트
        self.ids = filtered_img_ids
        print(f"반려된 어노테이션을 제외한 후 이미지 수: {len(self.ids)}")
    
    def _apply_annotation_weights(self):
        """반려된 어노테이션에 가중치 적용"""
        if not self.rejected_ids:
            return
        
        # 각 어노테이션에 가중치 적용
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                ann_id = ann['id']
                # 반려된 어노테이션에 더 높은 가중치 적용
                if ann_id in self.rejected_ids:
                    if ann.get('rejection_type') == 'hard_negative':
                        self.annotation_weights[ann_id] = self.hard_negative_weight
                    else:
                        self.annotation_weights[ann_id] = self.misclassified_weight
                else:
                    self.annotation_weights[ann_id] = 1.0
    
    def get_target(self, img_id):
        """타겟 데이터 가져오기 (가중치 정보 포함)"""
        target = super().get_target(img_id)
        
        # rejection_mode가 'weight'인 경우 가중치 정보 추가
        if self.rejection_mode == 'weight':
            # 어노테이션 ID 목록 가져오기
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            
            # 가중치 목록 생성
            weights = []
            for ann_id in ann_ids:
                weights.append(self.annotation_weights.get(ann_id, 1.0))
            
            # 가중치 정보 타겟에 추가
            target['weights'] = torch.tensor(weights, dtype=torch.float32)
        
        return target

def weighted_loss_function(original_loss_dict, targets):
    """
    가중치가 적용된 손실 함수
    """
    weighted_loss_dict = {}
    
    # 각 손실에 가중치 적용
    for key, loss in original_loss_dict.items():
        if 'weights' in targets:
            # 가중치 정보가 있는 경우 적용
            weights = targets['weights']
            # 가중치 평균 계산
            avg_weight = weights.mean().item()
            # 손실에 가중치 적용
            weighted_loss = loss * avg_weight
            weighted_loss_dict[key] = weighted_loss
        else:
            # 가중치 정보가 없는 경우 원래 손실 사용
            weighted_loss_dict[key] = loss
    
    return weighted_loss_dict

def train_one_epoch_with_weights(model, optimizer, data_loader, device, epoch, args):
    """
    가중치를 적용한 학습 함수 (원래 함수의 수정 버전)
    """
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = pmr.utils.Meter("total")
    m_m = pmr.utils.Meter("model")
    b_m = pmr.utils.Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
                   
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()
        
        losses = model(image, target)
        
        # 가중치 기반 손실 계산 적용
        if 'weights' in target:
            losses = weighted_loss_function(losses, target)
            
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print(f"\ndevice: {device}")
    
    # 반려된 어노테이션 ID 로드
    rejected_ids = load_rejected_annotations()
    print(f"반려된 어노테이션 ID 수: {len(rejected_ids)}")
    
    # 커스텀 데이터셋으로 학습 데이터 준비
    dataset_train = XylemDatasetWithRejected(
        args.data_dir, "train", train=True, 
        rejected_ids=rejected_ids, 
        rejection_mode=args.rejection_mode
    )
    
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    
    # 검증 데이터셋 - 기본 데이터셋 사용
    d_test = pmr.datasets("xylem", args.data_dir, "val", train=True)
    
    args.warmup_iters = max(1000, len(d_train))
    
    print(args)
    # +1 for include background class
    num_classes = max(d_train.dataset.classes) + 1 
    # ResNet-50 backbone Mask R-CNN Model create
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    
    # 가중치 로드
    if not args.no_pretrained:
        pretrained_path = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_coco_pretrained.pth")
        
        if not os.path.exists(pretrained_path):
            print("Downloading COCO pretrained weights...")
            import urllib.request
            url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
            urllib.request.urlretrieve(url, pretrained_path)
        
        print("Loading COCO pretrained weights...")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and "roi_heads.box_predictor" not in k 
                          and "roi_heads.mask_predictor" not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Pretrained weights loaded successfully!")
    
    # 옵티마이저 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # 체크포인트 찾기 및 최신 체크포인트 로드
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print(f"\nalready trained: {start_epoch} epochs; to {args.epochs} epochs")
    
    # 학습 시작
    for epoch in range(start_epoch, args.epochs):
        print(f"\nepoch: {epoch + 1}")
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print(f"lr_epoch: {args.lr_epoch:.5f}, factor: {lr_lambda(epoch):.5f}")
        
        # 커스텀 train_one_epoch 함수 (가중치 적용)
        if args.rejection_mode == 'weight':
            # 가중치를 적용한 train_one_epoch 함수 사용
            iter_train = train_one_epoch_with_weights(model, optimizer, d_train, device, epoch, args)
        else:
            # 기본 train_one_epoch 함수 사용
            iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        
        A = time.time() - A
        
        B = time.time()
        eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print(f"training: {A:.1f} s, evaluation: {B:.1f} s")
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        
        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, 
                      eval_info=str(eval_output))

        # 체크포인트 관리
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system(f"rm {ckpts[i]}")
    
    print(f"\ntotal time of this training: {time.time() - since:.1f} s")
    if start_epoch < args.epochs:
        print(f"already trained: {trained_epoch} epochs\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="반려된 어노테이션을 고려한 Mask R-CNN 훈련")
    
    # 기본 학습 파라미터
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float, default=0.01)  # 파인 튜닝에 적합한 학습률
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    
    # 데이터 및 모델 파라미터
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="xylem", help="coco or voc or xylem")
    parser.add_argument("--data-dir", default="/gdrive/MyDrive/HyunsLab/Xylemrcnn/dataset")
    parser.add_argument("--ckpt-path", default="./maskrcnn_xylem_rejected.pth")
    parser.add_argument("--results", default="./maskrcnn_rejected_results.pth")
    parser.add_argument("--no-pretrained", action="store_true", help="do not load pretrained weights")
    
    # 반려된 어노테이션 처리 방식
    parser.add_argument("--rejection-mode", choices=["exclude", "weight"], default="weight",
                       help="exclude: 반려된 어노테이션 제외, weight: 반려된 어노테이션에 가중치 적용")
    parser.add_argument("--hard-negative-weight", type=float, default=2.0,
                       help="거짓 양성(False Positive) 샘플에 적용할 손실 가중치")
    parser.add_argument("--misclassified-weight", type=float, default=3.0,
                       help="오분류된 샘플에 적용할 손실 가중치")
    
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.01  # 파인 튜닝을 위한 더 낮은 학습률 설정
    if args.ckpt_path is None:
        args.ckpt_path = f"./maskrcnn_{args.dataset}_rejected.pth"
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_rejected_results.pth")
    
    main(args)
