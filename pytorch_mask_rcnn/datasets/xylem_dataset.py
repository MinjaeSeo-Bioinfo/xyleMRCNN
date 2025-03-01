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
        
        # 이미지 및 어노테이션 경로 설정
        self.img_dir = os.path.join(data_dir, split)
        ann_file = os.path.join(data_dir, "annotations", f"result_{split}.json")
        
        # COCO API 초기화
        self.coco = COCO(ann_file)
        
        # 이미지 ID 가져오기
        img_ids = sorted([int(img_id) for img_id in self.coco.getImgIds()])
        print(f"이미지 ID 범위: {min(img_ids)}부터 {max(img_ids)}까지, 총 {len(img_ids)}개")
        self.ids = img_ids  # 명확하게 정수 리스트로 저장
        
        # ID 타입 확인
        if len(self.ids) > 0:
            print(f"self.ids의 첫 번째 ID 타입: {type(self.ids[0])}")
            print(f"self.ids의 첫 번째 값: {self.ids[0]}")
        
        # 유효한 ID 범위 확인
        coco_img_ids = set(self.coco.imgs.keys())
        print(f"COCO 이미지 ID 범위: {min(coco_img_ids)}부터 {max(coco_img_ids)}까지")
        
        # 클래스 정의 (배경은 0)
        self.classes = [0]
        self.classes.extend(sorted(self.coco.cats.keys()))

        # 데이터셋 체크
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
            
        self.ids = [int(id) for id in self.ids]
        
        # 확인 출력
        if len(self.ids) > 0:
            print(f"check_dataset 이후 self.ids의 첫 번째 ID 타입: {type(self.ids[0])}")
            print(f"check_dataset 이후 self.ids의 첫 번째 값: {self.ids[0]}")
                
        # 카테고리 매핑 생성
        cats = list(self.coco.cats.values())
        self.cat_ids = {cat['id']: i+1 for i, cat in enumerate(cats)}

        # 카테고리 정보 출력
        print("Categories:")
        for cat in cats:
            print(f"- {cat['name']} (ID: {cat['id']})")
        
        # debugging !
        print(f"self.ids의 첫 번째 ID 타입: {type(self.ids[0])}")
        print(f"self.coco.imgs의 첫 번째 키 타입: {type(next(iter(self.coco.imgs.keys())))}")

    def get_image(self, img_id):
        # 이미지 ID 타입 확인 및 변환
        if isinstance(img_id, str):
            img_id = int(img_id)
            
        # 이미지 ID가 데이터셋에 존재하는지 확인
        if img_id not in self.coco.imgs:
            raise KeyError(f"이미지 ID {img_id}가 데이터셋에 존재하지 않습니다. 유효한 ID 범위: {min(self.ids)}~{max(self.ids)}")
        
        img_info = self.coco.imgs[img_id]
        image_path = os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"])
        
        # 파일 존재 여부 확인
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
        image = Image.open(image_path)
        return image.convert("RGB")
    
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
        # 이미지 ID 타입 확인 및 변환
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

    def visualize_sample(self, idx):
        """Visualize a sample of a specific index"""
        image, target = self[idx]
        visualize_masks(image, target)
        
    def __getitem__(self, idx):
        if idx >= len(self.ids):
            raise IndexError(f"인덱스 {idx}가 데이터셋 크기({len(self.ids)})를 초과했습니다.")
        
        # self.ids[idx]가 여전히 문자열이라면 정수로 변환
        if isinstance(self.ids[idx], str):
            img_id = int(self.ids[idx])
        else:
            img_id = self.ids[idx]
        
        print(f"인덱스 {idx}에 대한 이미지 ID: {img_id}, 타입: {type(img_id)}")
        
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
