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
    """Load list of rejected annotation IDs"""
    rejected_ids = []
    if os.path.exists('rejected_annotations.txt'):
        with open('rejected_annotations.txt', 'r') as f:
            rejected_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    return rejected_ids

class XylemDatasetWithRejected(pmr.datasets.XylemDataset):
    """
    Dataset that excludes rejected annotations or applies weights to them
    """
    def __init__(self, data_dir, split, train=False, rejected_ids=None, rejection_mode='exclude', 
                 hard_negative_weight=2.0, misclassified_weight=3.0):
        super().__init__(data_dir, split, train)
        
        # Rejected annotation IDs
        self.rejected_ids = rejected_ids if rejected_ids is not None else []
        self.rejection_mode = rejection_mode  # 'exclude' or 'weight'
        
        # Weight settings
        self.hard_negative_weight = hard_negative_weight  # False Positive weight
        self.misclassified_weight = misclassified_weight  # Misclassification weight
        
        # Annotation weight map (used when rejection_mode='weight')
        self.annotation_weights = {}
        
        # Process rejected annotations
        if rejection_mode == 'exclude':
            # Filter dataset to exclude rejected annotation IDs
            self._filter_rejected_annotations()
        elif rejection_mode == 'weight':
            # Apply weights to rejected annotations
            self._apply_annotation_weights()
    
    def _filter_rejected_annotations(self):
        """Filter dataset to exclude rejected annotation IDs"""
        if not self.rejected_ids:
            return
        
        # Find image IDs to filter
        filtered_img_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Check if image has any rejected annotations
            has_rejected = any(ann['id'] in self.rejected_ids for ann in anns)
            
            # Include only images without rejected annotations
            if not has_rejected:
                filtered_img_ids.append(img_id)
        
        # Update dataset with filtered image IDs
        self.ids = filtered_img_ids
        print(f"Number of images after excluding rejected annotations: {len(self.ids)}")
    
    def _apply_annotation_weights(self):
        """Apply weights to rejected annotations"""
        if not self.rejected_ids:
            return
        
        # Apply weights to each annotation
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                ann_id = ann['id']
                # Apply higher weights to rejected annotations
                if ann_id in self.rejected_ids:
                    if ann.get('rejection_type') == 'hard_negative':
                        self.annotation_weights[ann_id] = self.hard_negative_weight
                    else:
                        self.annotation_weights[ann_id] = self.misclassified_weight
                else:
                    self.annotation_weights[ann_id] = 1.0
    
    def get_target(self, img_id):
        """Get target data (including weight information)"""
        target = super().get_target(img_id)
        
        # Add weight information if rejection_mode is 'weight'
        if self.rejection_mode == 'weight':
            # Get list of annotation IDs
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            
            # Create list of weights
            weights = []
            for ann_id in ann_ids:
                weights.append(self.annotation_weights.get(ann_id, 1.0))
            
            # Add weight information to target
            target['weights'] = torch.tensor(weights, dtype=torch.float32)
        
        return target

def weighted_loss_function(original_loss_dict, targets):
    """
    Loss function with applied weights
    """
    weighted_loss_dict = {}
    
    # Apply weights to each loss
    for key, loss in original_loss_dict.items():
        if 'weights' in targets:
            # Apply weights if weight information exists
            weights = targets['weights']
            # Calculate average weight
            avg_weight = weights.mean().item()
            # Apply weight to loss
            weighted_loss = loss * avg_weight
            weighted_loss_dict[key] = weighted_loss
        else:
            # Use original loss if no weight information
            weighted_loss_dict[key] = loss
    
    return weighted_loss_dict

def train_one_epoch_with_weights(model, optimizer, data_loader, device, epoch, args):
    """
    Training function with applied weights (modified version of original function)
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
        
        # Apply weight-based loss calculation
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
    
    # Load rejected annotation IDs
    rejected_ids = load_rejected_annotations()
    print(f"Number of rejected annotation IDs: {len(rejected_ids)}")
    
    # Prepare training data with custom dataset
    dataset_train = XylemDatasetWithRejected(
        args.data_dir, "train", train=True, 
        rejected_ids=rejected_ids, 
        rejection_mode=args.rejection_mode
    )
    
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    
    # Validation dataset - using default dataset
    d_test = pmr.datasets("xylem", args.data_dir, "val", train=True)
    
    args.warmup_iters = max(1000, len(d_train))
    
    print(args)
    # +1 for include background class
    num_classes = max(d_train.dataset.classes) + 1 
    # ResNet-50 backbone Mask R-CNN Model create
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    
    # Load weights
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
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # Find checkpoints and load latest checkpoint
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
    
    # Start training
    for epoch in range(start_epoch, args.epochs):
        print(f"\nepoch: {epoch + 1}")
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print(f"lr_epoch: {args.lr_epoch:.5f}, factor: {lr_lambda(epoch):.5f}")
        
        # Custom train_one_epoch function (with weights)
        if args.rejection_mode == 'weight':
            # Use train_one_epoch function with weights
            iter_train = train_one_epoch_with_weights(model, optimizer, d_train, device, epoch, args)
        else:
            # Use default train_one_epoch function
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

        # Checkpoint management
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
    parser = argparse.ArgumentParser(description="Mask R-CNN training with rejected annotations")
    
    # Basic training parameters
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float, default=0.01)  # Learning rate suitable for fine-tuning
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    
    # Data and model parameters
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="xylem", help="coco or voc or xylem")
    parser.add_argument("--data-dir", default="/gdrive/MyDrive/HyunsLab/Xylemrcnn/dataset")
    parser.add_argument("--ckpt-path", default="./maskrcnn_xylem_rejected.pth")
    parser.add_argument("--results", default="./maskrcnn_rejected_results.pth")
    parser.add_argument("--no-pretrained", action="store_true", help="do not load pretrained weights")
    
    # Rejected annotation handling method
    parser.add_argument("--rejection-mode", choices=["exclude", "weight"], default="weight",
                       help="exclude: exclude rejected annotations, weight: apply weights to rejected annotations")
    parser.add_argument("--hard-negative-weight", type=float, default=2.0,
                       help="Loss weight to apply to False Positive samples")
    parser.add_argument("--misclassified-weight", type=float, default=3.0,
                       help="Loss weight to apply to misclassified samples")
    
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.01  # Lower learning rate for fine-tuning
    if args.ckpt_path is None:
        args.ckpt_path = f"./maskrcnn_{args.dataset}_rejected.pth"
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_rejected_results.pth")
    
    main(args)
