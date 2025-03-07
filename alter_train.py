import sys
import bisect
import glob
import os
import re
import time
import torch
import pytorch_mask_rcnn as pmr
import argparse
from torch.utils.data import DataLoader
from pytorch_mask_rcnn.datasets.utils import collate_wrapper

BASE_DIR = '/gdrive/MyDrive/HyunsLab/Xylemrcnn'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    # ---------------------- prepare data loader ------------------------------- #
    print(f"Dataset argument received: {args.dataset}")
     
    # 하나의 데이터셋을 두 모델이 공유
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train", train=True)
    
    # 전체 데이터셋을 섞고 두 개의 동일한 크기의 부분으로 나눔
    indices = torch.randperm(len(dataset_train)).tolist()
    split_point = len(indices) // 2
    
    # Mask R-CNN용 데이터셋
    maskrcnn_indices = indices[:split_point]
    d_train_maskrcnn = torch.utils.data.Subset(dataset_train, maskrcnn_indices)
    
    # U-Net용 데이터셋
    unet_indices = indices[split_point:]
    d_train_unet = torch.utils.data.Subset(dataset_train, unet_indices)
    
    # 두 개의 데이터로더 생성
    maskrcnn_loader = DataLoader(
        d_train_maskrcnn, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_wrapper
    )
    
    unet_loader = DataLoader(
        d_train_unet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_wrapper
    )
    
    # 검증 데이터셋
    d_test = pmr.datasets(args.dataset, args.data_dir, "val", train=True)
    
    args.warmup_iters = max(1000, len(d_train_maskrcnn))
    
    # -------------------------------------------------------------------------- #
    
    print(args)
    # +1 for include background class
    num_classes = max(dataset_train.classes) + 1 
    
    # ---------------------- Create models ------------------------------------- #
    # 모델 임포트 - 직접 파일에서 임포트
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 현재 스크립트 경로 추가
    
    # 직접 임포트 시도
    try:
        from pytorch_mask_rcnn.model.mask_rcnn import maskrcnn_se_resnet50
    except ImportError:
        print("maskrcnn_se_resnet50 임포트 실패, 대안 사용")
        from pytorch_mask_rcnn.model.mask_rcnn import MaskRCNN
        from pytorch_mask_rcnn.model.SENet import SEResBackbone
        
        def maskrcnn_se_resnet50(pretrained, num_classes, boundary_weight=2.0):
            backbone = SEResBackbone('resnet50', pretrained_backbone=False)
            model = MaskRCNN(backbone, num_classes, boundary_weight=boundary_weight)
            return model
            
    from pytorch_mask_rcnn.model.UNet import XylemUNet, XylemUNetLoss
    
    # 1. Mask R-CNN 모델 생성
    maskrcnn = maskrcnn_se_resnet50(True, num_classes, boundary_weight=args.boundary_weight).to(device)
    
    # 2. U-Net 모델 생성
    unet = XylemUNet(n_channels=3, n_classes=1, with_attention=True).to(device)
    
    # COCO pretrained weight download for Mask R-CNN
    if not hasattr(args, "no_pretrained"):
        pretrained_path = os.path.join(os.path.dirname(args.maskrcnn_ckpt_path), "maskrcnn_coco_pretrained.pth")
        
        if not os.path.exists(pretrained_path):
            print("Downloading COCO pretrained weights...")
            import urllib.request
            url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
            urllib.request.urlretrieve(url, pretrained_path)
        
        print("Loading COCO pretrained weights for Mask R-CNN...")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = maskrcnn.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and "roi_heads.box_predictor" not in k 
                          and "roi_heads.mask_predictor" not in k}
        
        model_dict.update(pretrained_dict)
        maskrcnn.load_state_dict(model_dict)
        print("Pretrained weights loaded successfully!")
    
    # ---------------------- Create optimizers --------------------------------- #
    # Mask R-CNN 옵티마이저
    maskrcnn_params = [p for p in maskrcnn.parameters() if p.requires_grad]
    maskrcnn_optimizer = torch.optim.SGD(
        maskrcnn_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # U-Net 옵티마이저 (Adam 사용)
    unet_optimizer = torch.optim.Adam(
        unet.parameters(), lr=args.lr * 0.1)
    
    # 학습률 스케줄러
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    maskrcnn_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        maskrcnn_optimizer, milestones=args.lr_steps, gamma=0.1)
    unet_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        unet_optimizer, mode='min', patience=3, factor=0.5)
    
    # U-Net 손실 함수
    unet_criterion = XylemUNetLoss(boundary_weight=args.boundary_weight)
    
    # ---------------------- Load checkpoints ---------------------------------- #
    start_epoch = 0
    
    # Mask R-CNN 체크포인트 로드
    prefix, ext = os.path.splitext(args.maskrcnn_ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    if ckpts:
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        checkpoint = torch.load(ckpts[-1], map_location=device)
        maskrcnn.load_state_dict(checkpoint["model"])
        maskrcnn_optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        print(f"Loaded Mask R-CNN checkpoint from epoch {start_epoch}")
        del checkpoint
        torch.cuda.empty_cache()
    
    # U-Net 체크포인트 로드
    prefix, ext = os.path.splitext(args.unet_ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    if ckpts:
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        checkpoint = torch.load(ckpts[-1], map_location=device)
        unet.load_state_dict(checkpoint["model"])
        unet_optimizer.load_state_dict(checkpoint["optimizer"])
        unet_epoch = checkpoint["epochs"]
        print(f"Loaded U-Net checkpoint from epoch {unet_epoch}")
        # 두 모델의 에포크 동기화
        start_epoch = max(start_epoch, unet_epoch)
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
    unet_val_loss_history = []
    
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
        
        # 각 모델의 에포크 시작 시간 기록
        maskrcnn_start = time.time()
        
        # Mask R-CNN 훈련
        print("Training Mask R-CNN...")
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("Mask R-CNN lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        
        # engine.py의 train_one_epoch_custom 함수 사용
        iter_train_maskrcnn = pmr.train_one_epoch_custom(maskrcnn, maskrcnn_optimizer, maskrcnn_loader, device, epoch, args)
        maskrcnn_train_time = time.time() - maskrcnn_start
        
        # U-Net 훈련 시작
        unet_start = time.time()
        print("Training U-Net...")
        # U-Net은 Adam 옵티마이저를 사용하므로 학습률 직접 설정
        for param_group in unet_optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1 * lr_lambda(epoch)
        print("U-Net lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr * 0.1 * lr_lambda(epoch), lr_lambda(epoch)))
        
        # U-Net 훈련
        iter_train_unet = train_unet(unet, unet_optimizer, unet_criterion, unet_loader, device, epoch, args)
        unet_train_time = time.time() - unet_start
        
        # 평가 시작
        # Mask R-CNN 평가 - 수정된 evaluate_custom 함수 사용
        print("Evaluating Mask R-CNN...")
        maskrcnn_eval_start = time.time()
        eval_output_maskrcnn, iter_eval_maskrcnn = evaluate_custom(maskrcnn, d_test, device, args)
        maskrcnn_eval_time = time.time() - maskrcnn_eval_start
        
        # U-Net 평가
        print("Evaluating U-Net...")
        unet_eval_start = time.time()
        unet_val_loss = validate_unet(unet, unet_criterion, d_test, device)
        unet_val_loss_history.append(unet_val_loss)
        unet_eval_time = time.time() - unet_eval_start

        trained_epoch = epoch + 1
        print("Mask R-CNN - training: {:.1f} s, evaluation: {:.1f} s".format(maskrcnn_train_time, maskrcnn_eval_time))
        print("U-Net - training: {:.1f} s, evaluation: {:.1f} s".format(unet_train_time, unet_eval_time))
        
        # GPU 정보 수집
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train_maskrcnn, 1 / iter_eval_maskrcnn])
        
        # 체크포인트 저장
        # Mask R-CNN 체크포인트 - pmr.save_ckpt 사용
        pmr.save_ckpt(maskrcnn, maskrcnn_optimizer, trained_epoch, args.maskrcnn_ckpt_path, 
                      eval_info=str(eval_output_maskrcnn))
        
        # U-Net 체크포인트
        save_unet_ckpt(unet, unet_optimizer, trained_epoch, args.unet_ckpt_path, 
                      loss_info=unet_val_loss)

        # 학습률 스케줄러 단계
        maskrcnn_lr_scheduler.step()
        unet_lr_scheduler.step(unet_val_loss)
        
        # 체크포인트 파일 관리 (최신 10개만 유지)
        # Mask R-CNN 체크포인트
        prefix, ext = os.path.splitext(args.maskrcnn_ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
        # U-Net 체크포인트
        prefix, ext = os.path.splitext(args.unet_ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))

# U-Net 훈련 함수
def train_unet(model, optimizer, criterion, data_loader, device, epoch, args):
    """
    U-Net 모델 훈련
    """
    model.train()
    iters = len(data_loader) if args.iters < 0 else args.iters
    
    t_m = pmr.utils.Meter("total")
    m_m = pmr.utils.Meter("model")
    b_m = pmr.utils.Meter("backward")
    
    total_loss = 0.0
    processed_batches = 0
    A = time.time()
    
    for i, batch_data in enumerate(data_loader):
        if i >= iters:
            break
            
        T = time.time()
        
        # 데이터 준비
        if isinstance(batch_data, list) and len(batch_data) == 2:
            images, targets = batch_data
        else:
            try:
                images = batch_data.images
                targets = batch_data.targets
            except AttributeError:
                print(f"Unexpected batch format: {type(batch_data)}")
                continue
        
        # 배치 이미지 스택
        images = torch.stack([img.to(device) for img in images])
        
        # 마스크 준비
        masks = []
        for t in targets:
            if "masks" in t and t["masks"].numel() > 0:
                # 모든 인스턴스 마스크 결합
                instance_masks = t["masks"].to(device)
                combined_mask = torch.any(instance_masks > 0.5, dim=0).float()
                masks.append(combined_mask)
            else:
                # 빈 마스크 처리
                h, w = images[0].shape[-2:]
                masks.append(torch.zeros((h, w), device=device))
        
        # 마스크 텐서 준비
        masks = torch.stack(masks).unsqueeze(1)  # [B, 1, H, W]
        
        # U-Net 훈련
        S = time.time()
        optimizer.zero_grad()
        seg_pred, boundary_pred = model(images)
        loss, loss_dict = criterion(seg_pred, boundary_pred, masks)
        m_m.update(time.time() - S)
        
        S = time.time()
        loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        
        # 손실 출력
        if i % args.print_freq == 0:
            print(f"U-Net [{i}/{iters}], loss: {batch_loss:.4f}")
        
        t_m.update(time.time() - T)
        processed_batches += 1
    
    A = time.time() - A
    
    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        print("U-Net - iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}, avg_loss: {:.4f}".format(
            1000*A/processed_batches, 1000*t_m.avg, 1000*m_m.avg, 1000*b_m.avg, avg_loss))
    else:
        print("No U-Net batches were processed")
    
    return A / max(processed_batches, 1)

# U-Net 검증 함수
def validate_unet(model, criterion, data_loader, device):
    """
    U-Net 모델 검증
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            # 데이터 준비
            if isinstance(batch_data, list) and len(batch_data) == 2:
                images, targets = batch_data
            else:
                try:
                    images = batch_data.images
                    targets = batch_data.targets
                except AttributeError:
                    print(f"Unexpected batch format: {type(batch_data)}")
                    continue
            
            # 배치 이미지 스택
            images = torch.stack([img.to(device) for img in images])
            
            # 마스크 준비
            masks = []
            for t in targets:
                if "masks" in t and t["masks"].numel() > 0:
                    # 모든 인스턴스 마스크 결합
                    instance_masks = t["masks"].to(device)
                    combined_mask = torch.any(instance_masks > 0.5, dim=0).float()
                    masks.append(combined_mask)
                else:
                    # 빈 마스크 처리
                    h, w = images[0].shape[-2:]
                    masks.append(torch.zeros((h, w), device=device))
            
            # 마스크 텐서 준비
            if len(masks) > 0:
                masks = torch.stack(masks).unsqueeze(1)  # [B, 1, H, W]
                
                # U-Net 예측
                seg_pred, boundary_pred = model(images)
                loss, _ = criterion(seg_pred, boundary_pred, masks)
                
                total_loss += loss.item()
                num_batches += 1
    
    # 평균 손실 계산
    avg_loss = total_loss / max(num_batches, 1)  # 분모가 0이 되지 않도록 함
    print(f"U-Net validation loss: {avg_loss:.4f}")
    
    return avg_loss

# U-Net 체크포인트 저장 함수
def save_unet_ckpt(model, optimizer, epochs, ckpt_path, loss_info):
    """
    U-Net 모델 체크포인트 저장
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epochs": epochs,
        "loss": loss_info
    }
    
    prefix, ext = os.path.splitext(ckpt_path)
    epoch_ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, epoch_ckpt_path)
    
    # 최신 체크포인트 복사
    torch.save(checkpoint, ckpt_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask R-CNN과 U-Net 병렬 훈련")
    
    # 데이터 관련 인자
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aspect-ratio-group-factor", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true", help="do not load pretrained weights")
    
    # 훈련 관련 인자
    parser.add_argument("--boundary-weight", type=float, default=2.0,
                       help="Weight factor for boundary pixels in mask loss")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    
    # 기타 인자
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="xylem", help="coco or voc or xylem")
    parser.add_argument("--data-dir", default=DATASET_DIR)
    
    # 모델 체크포인트 경로
    parser.add_argument("--maskrcnn-ckpt-path", default=os.path.join(CHECKPOINT_DIR, "maskrcnn_xylem.pth"))
    parser.add_argument("--unet-ckpt-path", default=os.path.join(CHECKPOINT_DIR, "unet_xylem.pth"))
    parser.add_argument("--results", default=os.path.join(CHECKPOINT_DIR, "maskrcnn_results.pth"))

    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * args.batch_size / 16  # lr should be 'batch_size / 16 * 0.02'
    
    os.makedirs(os.path.dirname(args.maskrcnn_ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.unet_ckpt_path), exist_ok=True)

    main(args)

