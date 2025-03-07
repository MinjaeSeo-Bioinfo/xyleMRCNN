import sys
import time
import torch
from .utils import Meter, TextArea
from .datasets.xylem_eval import XylemEvaluator, prepare_for_xylem_coco

def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
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
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0 or True:
            boundary_weight = model.head.boundary_weight if hasattr(model.head, 'boundary_weight') else "N/A"
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()), f"\tBoundary Weight: {boundary_weight}")

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)
    
    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = XylemEvaluator(dataset.coco, iou_types)
    results = torch.load(args.results, map_location="cpu")
    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))
    temp = sys.stdout
    sys.stdout = TextArea()
    coco_evaluator.summarize()
    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_xylem_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters

#-------------------------------------------------------------#

# xylem custom train_one_epoch function
def train_one_epoch_custom(model, optimizer, data_loader, device, epoch, args):
    """
    Modified train_one_epoch function that can handle CustomBatch objects
    """
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    
    for i, batch_data in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
        
        # Process CustomBatch object
        if hasattr(batch_data, 'images') and hasattr(batch_data, 'targets'):
            # If it's a CustomBatch object
            image = batch_data.images
            target = batch_data.targets
        elif isinstance(batch_data, tuple) and len(batch_data) == 2:
            # If it's a (image, target) tuple
            image, target = batch_data
        else:
            print(f"Unexpected batch format: {type(batch_data)}")
            continue
        
        # Print debugging info
        print(f"Image type: {type(image)}")
        
        # If the model expects a single tensor
        if isinstance(image, (list, tuple)) and len(image) > 0:
            # Using first image only
            # image_input = image[0].to(device)
            # Or move all images to device
            image_input = [img.to(device) for img in image]
        else:
            image_input = image.to(device)
        
        if isinstance(image, list):
            print(f"Image list length: {len(image)}")
            if len(image) > 0:
                print(f"First image type: {type(image[0])}, shape: {image[0].shape}")
        
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        
        S = time.time()
        try:
            losses = model(image_input, target)
            total_loss = sum(losses.values())
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            print(f"Image input type: {type(image_input)}")
            if isinstance(image_input, list):
                print(f"Image input list length: {len(image_input)}")
                if len(image_input) > 0:
                    print(f"First image in list shape: {image_input[0].shape}")
            else:
                print(f"Image input shape: {image_input.shape}")
            continue
        
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
    
def evaluate_custom(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results_custom(model, data_loader, device, args)
    
    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    # Using XylemEvaluator
    from pytorch_mask_rcnn.datasets.xylem_eval import XylemEvaluator, prepare_for_xylem_coco
    
    coco_evaluator = XylemEvaluator(dataset.coco, iou_types)
    results = torch.load(args.results, map_location="cpu")
    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))
    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()
    coco_evaluator.summarize()
    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
# generate results file   
@torch.no_grad()   
def generate_results_custom(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    
    for i, batch_data in enumerate(data_loader):
        T = time.time()
        
        # Process CustomBatch object
        if hasattr(batch_data, 'images') and hasattr(batch_data, 'targets'):
            # If it's a CustomBatch object
            image = batch_data.images
            target = batch_data.targets
        elif isinstance(batch_data, tuple) and len(batch_data) == 2:
            # If it's a (image, target) tuple
            image, target = batch_data
        else:
            print(f"Unexpected batch format: {type(batch_data)}")
            continue
        
        image = [img.to(device) for img in image]
        target = {k: v.to(device) for k, v in target[0].items()}  # Using only first target

        S = time.time()
        output = model(image)
        m_m.update(time.time() - S)
        
        # Convert to Xylem format
        from pytorch_mask_rcnn.datasets.xylem_eval import prepare_for_xylem_coco
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_xylem_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters
