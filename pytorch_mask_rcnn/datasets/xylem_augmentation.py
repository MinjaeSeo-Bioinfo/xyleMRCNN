import json
import os
import numpy as np
import torch
import cv2
from PIL import Image
import random
from pycocotools.coco import COCO
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

# 올려주신 코드의 함수들을 활용
from .xylem_transform import apply_albumentations_transforms, masks_from_polygons, visualize_masks

# 경로 설정
BASE_DIR = '/gdrive/MyDrive/HyunsLab/Xylemrcnn'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_JSON = os.path.join(DATASET_DIR, 'annotations', 'result_train.json')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'augmented')
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, 'images')
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'annotations', 'augmentation_train.json')

# 출력 디렉토리 생성
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'annotations'), exist_ok=True)

# 시드 설정
random.seed(77)
np.random.seed(77)
torch.manual_seed(77)

def augment_dataset(json_path, img_dir, output_img_dir, output_json_path, num_augmentations=20, visualize=False):
    # 기존 COCO JSON 파일 로드
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 카테고리 정보 업데이트 (xylem ID를 0에서 1로 변경)
    for category in coco_data['categories']:
        if category['id'] == 0:  # xylem 클래스인 경우
            category['id'] = 1
            print(f"Xylem 카테고리 ID를 1로 변경했습니다: {category['name']}")
    
    # 기존 어노테이션의 카테고리 ID도 업데이트
    for ann in coco_data['annotations']:
        if ann['category_id'] == 0:  # xylem 클래스인 경우
            ann['category_id'] = 1
    
    # 이미지 ID를 0에서 1로 시작하도록 변경
    img_id_mapping = {}  # 기존 ID -> 새 ID 매핑
    new_images = []
    
    for idx, img in enumerate(coco_data['images']):
        old_id = img['id']
        new_id = idx + 1  # 1부터 시작
        img_id_mapping[old_id] = new_id
        
        img_copy = img.copy()
        img_copy['id'] = new_id
        new_images.append(img_copy)
    
    # 어노테이션의 image_id 업데이트
    new_annotations = []
    for ann in coco_data['annotations']:
        old_img_id = ann['image_id']
        if old_img_id in img_id_mapping:
            ann_copy = ann.copy()
            ann_copy['image_id'] = img_id_mapping[old_img_id]
            new_annotations.append(ann_copy)
    
    # 업데이트된 데이터로 coco_data 갱신
    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations
    
    # 업데이트된 JSON 임시 저장
    temp_json_path = json_path + '.temp'
    with open(temp_json_path, 'w') as f:
        json.dump(coco_data, f)
    
    # 업데이트된 JSON으로 COCO API 초기화
    coco = COCO(temp_json_path)
    
    # 임시 파일 제거
    os.remove(temp_json_path)
    
    # 데이터 구조 복사
    new_coco_data = copy.deepcopy(coco_data)
    
    # 새로운 이미지와 어노테이션을 위한 ID 카운터 초기화
    max_img_id = max([img['id'] for img in coco_data['images']])
    max_ann_id = max([ann['id'] for ann in coco_data['annotations']])
    
    new_img_id = max_img_id + 1
    new_ann_id = max_ann_id + 1
    
    # 새로운 증강 이미지와 어노테이션 리스트 초기화
    aug_images = []
    aug_annotations = []
    
    # 모든 이미지 ID
    img_ids = coco.getImgIds()
    
    print(f"증강 시작: 원본 이미지 {len(img_ids)}개, 각 이미지당 {num_augmentations}개 증강")
    print(f"이미지 ID 범위: {min(img_ids)}부터 {max(img_ids)}까지")
    
    # 각 이미지에 대해 증강 수행
    for img_id in tqdm(img_ids):
        # 이미지 정보 로드
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # 나머지 기존 코드는 그대로 유지...
        # (이미지 로드, 어노테이션 로드, 증강 등)
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 어노테이션 로드
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 어노테이션에서 바운딩 박스, 클래스 라벨, 세그멘테이션 추출
        bboxes = []
        class_labels = []
        polygons = []
        
        for ann in anns:
            bbox = ann['bbox']
            if bbox[2] > 0 and bbox[3] > 0:
                bboxes.append(bbox)
                class_labels.append(ann['category_id'])
                polygons.append(ann['segmentation'])
        
        # 폴리곤을 마스크로 변환
        height, width = img_info['height'], img_info['width']
        masks = masks_from_polygons(polygons, height, width)
        
        # 각 이미지에 대해 여러 개의 증강 이미지 생성
        for aug_idx in range(num_augmentations):
            try:
                # 증강 적용
                aug_image, aug_masks, aug_bboxes, aug_labels = apply_albumentations_transforms(
                    image, masks, bboxes, class_labels, height, width
                )
                
                # 시각화 (필요한 경우)
                if visualize and aug_idx == 0:
                    tensor_masks = torch.from_numpy(np.array(aug_masks))
                    tensor_boxes = torch.tensor(aug_bboxes)
                    target = {'masks': tensor_masks, 'boxes': tensor_boxes}
                    visualize_masks(aug_image, target)
                
                # 증강된 이미지 파일명 생성
                base_name = os.path.splitext(img_info['file_name'])[0]
                ext = os.path.splitext(img_info['file_name'])[1]
                aug_filename = f"{base_name}_aug_{aug_idx}{ext}"
                
                # 증강된 이미지 저장
                aug_image.save(os.path.join(output_img_dir, aug_filename))
                
                # 새로운 이미지 정보 추가
                new_img_info = {
                    'id': new_img_id,
                    'file_name': aug_filename,
                    'height': height,
                    'width': width,
                    'license': img_info.get('license', 0),
                    'coco_url': img_info.get('coco_url', ''),
                    'date_captured': img_info.get('date_captured', '')
                }
                aug_images.append(new_img_info)
                
                # 증강된 어노테이션 추가
                for i, (bbox, mask, label) in enumerate(zip(aug_bboxes, aug_masks, aug_labels)):
                    # 마스크에서 폴리곤 생성
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = []
                    
                    for contour in contours:
                        contour = contour.flatten().tolist()
                        if len(contour) >= 6:  # 최소 3개의 점 필요
                            segmentation.append(contour)
                    
                    # 어노테이션이 비어있으면 건너뛰기
                    if not segmentation:
                        continue
                    
                    # 바운딩 박스 좌표 확인 및 수정
                    x, y, w, h = bbox
                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)
                    
                    # 너무 작은 바운딩 박스는 건너뛰기
                    if w < 5 or h < 5:
                        continue
                    
                    # 클래스 ID 변경
                    new_label = 1 if label == 0 else label
                    
                    # 새로운 어노테이션 추가
                    new_ann = {
                        'id': new_ann_id,
                        'image_id': new_img_id,
                        'category_id': new_label,
                        'segmentation': segmentation,
                        'area': float(w * h),
                        'bbox': [x, y, w, h],
                        'iscrowd': 0
                    }
                    aug_annotations.append(new_ann)
                    new_ann_id += 1
                
                new_img_id += 1
                
            except Exception as e:
                print(f"이미지 {img_info['file_name']} 증강 중 오류 발생: {e}")
                continue
    
    # 기존 데이터와 새로운 데이터 결합
    new_coco_data['images'].extend(aug_images)
    new_coco_data['annotations'].extend(aug_annotations)
    
    # 업데이트된 COCO JSON 파일 저장
    with open(output_json_path, 'w') as f:
        json.dump(new_coco_data, f)
    
    print(f"증강 완료: 원본 이미지 {len(img_ids)}개, 생성된 증강 이미지 {len(aug_images)}개")
    print(f"원본 어노테이션 {len(coco_data['annotations'])}개, 증강 어노테이션 {len(aug_annotations)}개")
    print(f"증강된 데이터셋이 {output_json_path}에 저장되었습니다.")
    
# 실행 예시
def main():
    # 이미지 디렉토리와 JSON 파일 경로 확인
    train_img_dir = os.path.join(DATASET_DIR, 'train')
    if not os.path.exists(train_img_dir):
        print(f"이미지 디렉토리가 존재하지 않습니다: {train_img_dir}")
        return
        
    if not os.path.exists(TRAIN_JSON):
        print(f"COCO JSON 파일이 존재하지 않습니다: {TRAIN_JSON}")
        return
    
    # 데이터셋 증강 실행
    augment_dataset(
        json_path=TRAIN_JSON,
        img_dir=train_img_dir,
        output_img_dir=OUTPUT_IMG_DIR,
        output_json_path=OUTPUT_JSON,
        num_augmentations=25,  # 20장을 500장 이상으로 만들기 위해 25배 증강 설정
        visualize=True  # 첫 번째 증강 결과 시각화 (확인용)
    )

if __name__ == "__main__":
    main()
