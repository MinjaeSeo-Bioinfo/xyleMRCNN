import json
import os
import numpy as np
import torch
import cv2
import copy
import matplotlib.pyplot as plt
import random
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from .xylem_transform import apply_albumentations_transforms, masks_from_polygons, visualize_masks

# Path configuration
BASE_DIR = '/gdrive/MyDrive/HyunsLab/Xylemrcnn'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_JSON = os.path.join(DATASET_DIR, 'annotations', 'result_train.json')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'augmented')
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, 'images')
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'annotations', 'augmentation_train.json')

# Create output directories
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'annotations'), exist_ok=True)

# Set random seed
random.seed(77)
np.random.seed(77)
torch.manual_seed(77)

def augment_dataset(json_path, img_dir, output_img_dir, output_json_path, num_augmentations=20, visualize=False):
    # Load existing COCO JSON file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Update category information
    for category in coco_data['categories']:
        if category['id'] == 0: 
            category['id'] = 1
            print(f"Changed Xylem category ID to 1: {category['name']}")
    
    # Change annotation ID to start from 1
    ann_id_mapping = {} 
    new_annotations = []
    
    for idx, ann in enumerate(coco_data['annotations']):
        old_id = ann['id']
        new_id = idx + 1
        ann_id_mapping[old_id] = new_id
        
        ann_copy = ann.copy()
        ann_copy['id'] = new_id
        
        # Also change category ID (existing code)
        if ann_copy['category_id'] == 0:
            ann_copy['category_id'] = 1
            
        new_annotations.append(ann_copy)
    
    # Change image ID to start from 1 
    img_id_mapping = {}
    new_images = []
    
    for idx, img in enumerate(coco_data['images']):
        old_id = img['id']
        new_id = idx + 1
        img_id_mapping[old_id] = new_id
        
        img_copy = img.copy()
        img_copy['id'] = new_id
        new_images.append(img_copy)
    
    # Update image_id in annotations
    for ann in new_annotations:
        old_img_id = ann['image_id']
        if old_img_id in img_id_mapping:
            ann['image_id'] = img_id_mapping[old_img_id]
    
    # Update coco_data with updated data
    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations
    
    # Temporary save of updated JSON
    temp_json_path = json_path + '.temp'
    with open(temp_json_path, 'w') as f:
        json.dump(coco_data, f)
    
    # Initialize COCO API with updated JSON
    coco = COCO(temp_json_path)
    
    # Remove temporary file
    os.remove(temp_json_path)
    
    # Copy data structure
    new_coco_data = copy.deepcopy(coco_data)
    
    # Calculate starting IDs for new augmentations
    max_img_id = max([img['id'] for img in coco_data['images']]) 
    max_ann_id = max([ann['id'] for ann in coco_data['annotations']])
    
    new_img_id = max_img_id + 1
    new_ann_id = max_ann_id + 1
    
    # Initialize lists for new augmented images and annotations
    aug_images = []
    aug_annotations = []
    
    # All image IDs
    img_ids = coco.getImgIds()
    
    print(f"Starting augmentation: {len(img_ids)} original images, {num_augmentations} augmentations per image")
    print(f"Image ID range: from {min(img_ids)} to {max(img_ids)}")
    
    # Perform augmentation for each image
    for img_id in tqdm(img_ids):
        # Load image information
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Extract bounding boxes, class labels, and segmentation from annotations
        bboxes = []
        class_labels = []
        polygons = []
        
        for ann in anns:
            bbox = ann['bbox']
            if bbox[2] > 0 and bbox[3] > 0:
                bboxes.append(bbox)
                class_labels.append(ann['category_id'])
                polygons.append(ann['segmentation'])
        
        # Convert polygons to masks
        height, width = img_info['height'], img_info['width']
        masks = masks_from_polygons(polygons, height, width)
        
        # Create multiple augmented images for each image
        for aug_idx in range(num_augmentations):
            try:
                # Apply augmentation
                aug_image, aug_masks, aug_bboxes, aug_labels = apply_albumentations_transforms(
                    image, masks, bboxes, class_labels, height, width
                )
                
                # Visualization
                if visualize and aug_idx == 0:
                    tensor_masks = torch.from_numpy(np.array(aug_masks))
                    tensor_boxes = torch.tensor(aug_bboxes)
                    target = {'masks': tensor_masks, 'boxes': tensor_boxes}
                    visualize_masks(aug_image, target)
                
                # Create filename for augmented image
                base_name = os.path.splitext(img_info['file_name'])[0]
                ext = os.path.splitext(img_info['file_name'])[1]
                aug_filename = f"{base_name}_aug_{aug_idx}{ext}"
                
                # Save augmented image
                aug_image.save(os.path.join(output_img_dir, aug_filename))
                
                # Add new image information
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
                
                # Add augmented annotations
                for i, (bbox, mask, label) in enumerate(zip(aug_bboxes, aug_masks, aug_labels)):
                    # Create polygons from masks
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = []
                    
                    for contour in contours:
                        contour = contour.flatten().tolist()
                        if len(contour) >= 6:
                            segmentation.append(contour)
                    
                    # Skip if annotation is empty
                    if not segmentation:
                        continue
                    
                    # Check and modify bounding box coordinates
                    x, y, w, h = bbox
                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)
                    
                    # Skip if bounding box is too small
                    if w < 5 or h < 5:
                        continue
                    
                    # Change class ID
                    new_label = 1 if label == 0 else label
                    
                    # Add new annotation
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
                print(f"Error occurred during augmentation of image {img_info['file_name']}: {e}")
                continue
    
    # Combine existing data with new data
    new_coco_data['images'].extend(aug_images)
    new_coco_data['annotations'].extend(aug_annotations)
    
    # Save updated COCO JSON file
    with open(output_json_path, 'w') as f:
        json.dump(new_coco_data, f)
    
    print(f"Augmentation completed: {len(img_ids)} original images, {len(aug_images)} augmented images generated")
    print(f"{len(coco_data['annotations'])} original annotations, {len(aug_annotations)} augmented annotations")
    print(f"Augmented dataset saved to {output_json_path}")
    
# Example execution
def main():
    # Check image directory and JSON file path
    train_img_dir = os.path.join(DATASET_DIR, 'train')
    if not os.path.exists(train_img_dir):
        print(f"Image directory does not exist: {train_img_dir}")
        return
        
    if not os.path.exists(TRAIN_JSON):
        print(f"COCO JSON file does not exist: {TRAIN_JSON}")
        return
    
    # Execute dataset augmentation
    augment_dataset(
        json_path=TRAIN_JSON,
        img_dir=train_img_dir,
        output_img_dir=OUTPUT_IMG_DIR,
        output_json_path=OUTPUT_JSON,
        num_augmentations=25,  # Set to 25x augmentation
        visualize=True  # Visualize the first augmentation result
    )

if __name__ == "__main__":
    main()
