import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from .generalized_dataset import GeneralizedDataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class XylemDataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        #@@
        self.img_dir = os.path.join(data_dir, split)
        ann_file = os.path.join(data_dir, "annotations", f"result_{split}.json")
        #@@
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = [0]  # 배경 클래스
        self.classes.extend(sorted(self.coco.cats.keys()))

        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
            
        #@ Create category mapping
        cats = list(self.coco.cats.values())
        self.cat_ids = {cat['id']: i+1 for i, cat in enumerate(cats)}

        #@ Print category information
        print("Categories:")
        for cat in cats:
            print(f"- {cat['name']} (ID: {cat['id']})")
            

    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"]))
        return image.convert("RGB")
        #@
        # print(f"Total images: {len(img_ids)}")
        
    def convert_to_xyxy(self, boxes):
        if boxes is None or len(boxes) == 0:
            return boxes
            
        # [x, y, width, height]를 [x1, y1, x2, y2]로 변환
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # [x1, y1, x2, y2] 형식으로 스택
        return torch.stack((x1, y1, x2, y2), dim=1)
    
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
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
            masks = torch.stack(masks)

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

    def visualize_sample(self, idx):
        """Visualize a sample of a specific index"""
        image, target = self[idx]
        visualize_masks(image, target)
        
    def __getitem__(self, idx):
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
