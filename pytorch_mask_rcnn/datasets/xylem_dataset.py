import os
import cv2
import json
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from .generalized_dataset import GeneralizedDataset

class XylemDataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        
        # Set data directory, split, and training mode
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        # Set image and annotation paths
        self.img_dir = os.path.join(data_dir, "augmented", "images", split)
        ann_file = os.path.join(data_dir, "augmented", "annotations", f"augmentation_{split}.json")
        
        # Initialize COCO API
        self.coco = COCO(ann_file)
        
        # Get image IDs
        img_ids = sorted([int(img_id) for img_id in self.coco.getImgIds()])
        print(f"Image ID range: from {min(img_ids)} to {max(img_ids)}, total {len(img_ids)} images")
        self.ids = img_ids

        # Check valid ID range
        coco_img_ids = set(self.coco.imgs.keys())
        print(f"COCO image ID range: from {min(coco_img_ids)} to {max(coco_img_ids)}")
        
        # Define classes (background is 0)
        self.classes = [0]  # Background class
        self.classes.extend(sorted(self.coco.cats.keys()))

        # Dataset check
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
        
        # Convert self.ids back to integers if changed to strings in check_dataset
        self.ids = [int(id) for id in self.ids]
                
        # Create category mapping
        cats = list(self.coco.cats.values())
        self.cat_ids = {cat['id']: i+1 for i, cat in enumerate(cats)}
            
    def get_image(self, img_id):
        # Check and convert image ID type
        if isinstance(img_id, str):
            img_id = int(img_id)
            
        # Check if image ID exists in dataset
        if img_id not in self.coco.imgs:
            raise KeyError(f"Image ID {img_id} does not exist in the dataset. Valid ID range: {min(self.ids)}~{max(self.ids)}")
        
        img_info = self.coco.imgs[img_id]
        image_path = os.path.join(self.img_dir, img_info["file_name"])
        
        # Check file existence
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = Image.open(image_path)
        return image.convert("RGB")
    
    def convert_to_xyxy(self, boxes):
        if boxes is None or len(boxes) == 0:
            return boxes
            
        # Convert [x, y, width, height] to [x1, y1, x2, y2]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Stack in [x1, y1, x2, y2] format
        return torch.stack((x1, y1, x2, y2), dim=1)
    
    def get_target(self, img_id):
        # Check and convert image ID type
        if isinstance(img_id, str):
            img_id = int(img_id)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels, dtype=torch.long)
            masks = torch.stack(masks) if len(masks) > 0 else torch.zeros((0, 0, 0), dtype=torch.uint8)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
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
            
            return torch.from_numpy(mask)
            
        except Exception as e:
            print(f"Error in polygon to mask conversion: {e}")
            print(f"Polygon shape: {np.array(polygons).shape}")
            return torch.zeros((height, width), dtype=torch.uint8)
 
    def __getitem__(self, idx):
        if idx >= len(self.ids):
            raise IndexError(f"Index {idx} exceeds dataset size ({len(self.ids)}).")
        
        # Convert self.ids[idx] to integer if it's still a string
        if isinstance(self.ids[idx], str):
            img_id = int(self.ids[idx])
        else:
            img_id = self.ids[idx]

        image = self.get_image(img_id)
        target = self.get_target(img_id)
        image = F.to_tensor(image)
    
        return image, target
        
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def convert_to_coco_api(ds):
        """Convert to COCO API format"""
        return ds.coco
