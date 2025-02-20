import bisect
import glob
import os
import re
import time
import torch
import pytorch_mask_rcnn as pmr
import argparse

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\\ndevice: {}".format(device))
    
    # ---------------------- prepare data loader ------------------------------- #
    # debugging - dataset work 
    print(f"Dataset argument received: {args.dataset}")
     
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val", train=True)
    
    args.warmup_iters = max(1000, len(d_train))
    
    # -------------------------------------------------------------------------- #
    
    print(args)
    # +1 for include background class
    num_classes = max(d_train.dataset.classes) + 1 
    # ResNet-50 backbone Mask R-CNN Model create
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    
    
    #@@@@@@@@ COCO pretrained weight download @@@@@@@@@@@
    if not hasattr(args, "no_pretrained"):
        
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
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    # params > set of trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # find all checkpoints, and load the latest checkpoint
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
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        # train_one_epoch from engine.py
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        
        #@ new codes for debugging
        ap_values = eval_output.get_AP()
        print(ap_values)
        
        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, 
                      eval_info=str(eval_output), ap_values=ap_values)

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
BASE_DIR = '/gdrive/MyDrive/HyunsLab/Xylemrcnn'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if __name__ == "__main__":
    # original factor
    parser = argparse.ArgumentParser()
    
    #@@@ new factor for data augmentation @@
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aspect-ratio-group-factor", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true", help="do not load pretrained weights")
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="xylem", help="coco or voc or xylem")
    parser.add_argument("--data-dir", default=DATASET_DIR)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")

    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
        
    args.results = "/gdrive/MyDrive/HyunsLab/Xylemrcnn/xyleMRCNN/maskrcnn_results.pth"    
    main(args)
