import numpy as np
import torch
import cv2
import random
import matplotlib.pyplot as plt
import albumentations as A
import torchvision.transforms.functional as F
import os
from PIL import Image

random.seed(77)
np.random.seed(77)
torch.manual_seed(77)

def visualize_masks(image, target):
    """
    visualize sample from data(transformed)
    
    Args:
        image (torch.Tensor or np.ndarray or PIL.Image): input image
        target (dict): target info dictionary
    """
    # Image transformation logic
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
    elif hasattr(image, 'convert'):  # PIL Image
        image = np.array(image)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax[0].imshow(image)
    ax[0].set_title("Transformed Image")
    
    # Draw bbox
    for box in target.get('boxes', []):
        x1, y1, x2, y2 = box.numpy()
        ax[0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
    
    # Draw mask
    masks = target.get('masks', [])
    if len(masks) > 0:
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        
        combined_mask = masks.sum(dim=0).numpy()
        ax[1].imshow(combined_mask)
        ax[1].set_title("Transformed Mask")
    
    plt.tight_layout()
    plt.show()

def apply_albumentations_transforms(image, masks, bboxes, class_labels, height, width):
    """
    Albumentations lib for data augmentation
    
    Args:
        image: PIL image
        masks: numpy mask array [n_instances, height, width]
        bboxes: COCO format bounding box [n_instances, 4] - [x_min, y_min, width, height]
        class_labels: class label list
        height, width: output image size
        
    Returns:
        transformed image, mask, bbox, label
    """
    # PIL image to numpy array
    image_np = np.array(image)
    
    # Transform
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        ], p=0.5),
        A.ISONoise(color_shift=(0.005, 0.02), intensity=(0.05, 0.15), p=0.2),
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.03, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.3),
        ], p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    ], bbox_params=A.BboxParams(
        format='coco',
        min_visibility=0.3,
        label_fields=['class_labels']
    ))
    
    # Apply the transformation
    try:
        transformed = transform(
            image=image_np,
            masks=masks if len(masks) > 0 else None,
            bboxes=bboxes if len(bboxes) > 0 else [],
            class_labels=class_labels
        )
        
        # Extract results
        transformed_image = transformed['image']
        transformed_masks = transformed['masks'] if 'masks' in transformed and transformed['masks'] is not None else []
        transformed_bboxes = transformed['bboxes'] if 'bboxes' in transformed else []
        transformed_labels = transformed['class_labels'] if 'class_labels' in transformed else []
        
        # Validation
        if len(transformed_bboxes) != len(transformed_labels):
            print(f"Warning: Mismatch between bboxes ({len(transformed_bboxes)}) and labels ({len(transformed_labels)})")
            
        # Convert to PIL images
        pil_image = Image.fromarray(transformed_image)
        
        return pil_image, transformed_masks, transformed_bboxes, transformed_labels
        
    except Exception as e:
        print(f"Error during transformation: {e}")
        return image, masks, bboxes, class_labels

def get_image(self, img_id):
    """Load an image from a COCO image ID"""
    img_info = self.coco.loadImgs(img_id)[0]
    file_name = img_info['file_name'].replace('\\', '/')
    if file_name.startswith('images/'):
        file_name = file_name.replace('images/', '')
    img_path = os.path.join(self.img_dir, file_name)
    return Image.open(img_path).convert('RGB')

def masks_from_polygons(polygons, height, width):
    """Convert COCO polygons to binary masks"""
    masks = []
    for polygon_group in polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for polygon in polygon_group:
            # Convert polygon points to an (N,2) array
            points = np.array(polygon).reshape(-1, 2)
            points = np.round(points).astype(np.int32)
            
            # Fill a polygonal region
            cv2.fillPoly(mask, [points], color=1)
            
        masks.append(mask)
    return np.array(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)

def get_annotations(self, img_id):
    """Loading annotations from COCO image IDs"""
    # Get iamge info
    img_info = self.coco.loadImgs(img_id)[0]
    height, width = img_info['height'], img_info['width']
    
    # Get annotations
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)
    
    bboxes = []
    class_labels = []
    polygons = []
    
    for ann in anns:
        # [x_min, y_min, width, height] format (bbox)
        bbox = ann['bbox']
        
        # only valid bbox
        if bbox[2] > 0 and bbox[3] > 0:
            bboxes.append(bbox)
            class_labels.append(self.cat_ids[ann['category_id']])
            polygons.append(ann['segmentation'])
    
    # poly2mask
    masks = masks_from_polygons(polygons, height, width)
    
    return {
        'bboxes': bboxes,
        'class_labels': class_labels,
        'masks': masks,
        'height': height,
        'width': width
    }
