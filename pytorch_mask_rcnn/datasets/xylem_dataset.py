import os
import cv2
import json
import numpy as np
import torch
import math
import random
from PIL import Image
from pycocotools.coco import COCO
from .generalized_dataset import GeneralizedDataset
from .xylem_dataset_utils import apply_albumentations_transforms, visualize_masks
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import albumentations as A

class XylemDataset(GeneralizedDataset):
    def __init__(self, data_dir, split='train', train=True):
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        # Define paths according to your structure
        self.img_dir = os.path.join(data_dir, split)
        ann_file = os.path.join(data_dir, "annotations", f"result_{split}.json")
        
        # Initialize COCO API properly
        self.coco = COCO(ann_file)
        
        # Get categories
        cats = self.coco.loadCats(self.coco.getCatIds())
        
        # Use COCO API methods
        img_ids = self.coco.getImgIds()
        ann_ids = self.coco.getAnnIds()
        
        print(f"Dataset: {split}")
        print(f"Total images: {len(img_ids)}")
        print(f"Total annotations: {len(ann_ids)}")
        
        # Prepare image IDs
        self.ids = list(sorted(img_ids))
        
        # Create category mapping
        self.cat_ids = {cat['id']: i+1 for i, cat in enumerate(cats)}
        
        # Add class information (including background)
        self.cat_names = ['background'] + [cat['name'] for cat in cats]
        self.classes = list(range(len(self.cat_names)))
        
        # Print category information
        print("Categories:")
        for cat in cats:
            print(f"- {cat['name']} (ID: {cat['id']})")
        
        # Calculate aspect ratios
        self._aspect_ratios = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            self._aspect_ratios.append(float(img_info['width']) / float(img_info['height']))

    def get_image(self, img_id):
        """Load and return image"""
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Clean up the file name
        file_name = img_info['file_name'].replace('\\', '/')
        if file_name.startswith('images/'):
            file_name = file_name.replace('images/', '')
        
        img_path = os.path.join(self.img_dir, file_name)
        return Image.open(img_path).convert('RGB')
    
    def get_target(self, img_id):
        target = {}
        target["image_id"] = torch.tensor([img_id])
        
        # Get annotations using COCO API
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Get image info
        img_info = self.coco.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']
        
        boxes = []
        masks = []
        labels = []
        
        for ann in anns:
            # COCO bbox format is [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Convert to [x1, y1, x2, y2] format and ensure within image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            
            # Only add valid boxes
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                
                if isinstance(ann['segmentation'], list):
                    mask = self._poly2mask(ann['segmentation'], height, width)
                    masks.append(mask)
                labels.append(self.cat_ids[ann['category_id']])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        target["boxes"] = boxes
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["masks"] = torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
        
        return target

    def _poly2mask(self, polygons, height, width):
        """Convert polygon to mask using OpenCV"""
        try:
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Convert polygon points
            for polygon in polygons:
                # Reshape polygon points to (N,2) array
                points = np.array(polygon).reshape(-1, 2)
                
                # Convert to integer coordinates
                points = np.round(points).astype(np.int32)
                
                # Fill polygon with white (255)
                cv2.fillPoly(mask, [points], color=255)
            
            # Convert to binary mask (0 or 1)
            mask = (mask > 0).astype(np.uint8)
            
            # Debug print
            print(f"Mask sum: {mask.sum()}, unique values: {np.unique(mask)}")
            
            return torch.from_numpy(mask)
            
        except Exception as e:
            print(f"Error in polygon to mask conversion: {e}")
            print(f"Polygon shape: {np.array(polygons).shape}")
            return torch.zeros((height, width), dtype=torch.uint8)
    
    def apply_transforms(self, image, target):
        if not self.train:
            return F.normalize(F.to_tensor(image), 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]), target
        
        # 1. Get original image
        h, w = image.size[1], image.size[0]
        
        # 2. Get data from target
        # Bbox format change, [x1, y1, x2, y2] -> [x, y, w, h]
        if len(target['boxes']):
            bboxes = []
            for box in target['boxes'].numpy():
                x1, y1, x2, y2 = box
                bboxes.append([x1, y1, x2-x1, y2-y1])
            
            # Convert mask to numpy array
            masks = target['masks'].numpy() if len(target['masks']) > 0 else []
            labels = target['labels'].numpy().tolist()
            
            # 3. Albumentations transform 
            pil_image, transformed_masks, transformed_bboxes, transformed_labels = apply_albumentations_transforms(
                image=image,
                masks=masks,
                bboxes=bboxes,
                class_labels=labels,
                height=h,
                width=w
            )
            
            # 4. Align the converted data back to the target
            # [x, y, w, h] -> [x1, y1, x2, y2] 
            if transformed_bboxes:
                final_boxes = []
                for box in transformed_bboxes:
                    x, y, w, h = box
                    final_boxes.append([x, y, x+w, y+h])
                
                target['boxes'] = torch.tensor(final_boxes, dtype=torch.float32)
                target['masks'] = torch.tensor(transformed_masks, dtype=torch.uint8)
                target['labels'] = torch.tensor(transformed_labels, dtype=torch.int64)
            else:
                # Generate an empty tensor if there are no boxes after conversion
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['masks'] = torch.zeros((0, h, w), dtype=torch.uint8)
                target['labels'] = torch.zeros(0, dtype=torch.int64)
            
            # 5. Convert an image to a tensor and normalize it
            image = F.to_tensor(pil_image)
            image = F.normalize(image,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            
            return image, target
        else:
            # Apply basic normalization if the box is not present
            image = F.to_tensor(image) 
            image = F.normalize(image,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        return image, target

    def visualize_sample(self, idx):
        """Visualize a sample of a specific index"""
        image, target = self[idx]
        visualize_masks(image, target)
        
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image = self.get_image(img_id)
        target = self.get_target(img_id)

        if self.train:
            image, target = self.apply_transforms(image, target)
        
        # Image should already be tensor after apply_transforms
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
            image = F.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        
        return image, target
        
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def convert_to_coco_api(ds):
        """Convert to COCO API format"""
        return ds.coco
