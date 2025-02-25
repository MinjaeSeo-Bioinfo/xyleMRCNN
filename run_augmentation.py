import os
import sys
from pytorch_mask_rcnn.datasets.xylem_augmentation import augment_dataset

def setup_directories(base_dir, split):
    """디렉토리 설정 및 생성"""
    dataset_dir = os.path.join(base_dir, 'dataset')
    input_json = os.path.join(dataset_dir, 'annotations', f'result_{split}.json')
    
    output_dir = os.path.join(dataset_dir, 'augmented')
    output_img_dir = os.path.join(output_dir, 'images', split)
    output_json = os.path.join(output_dir, 'annotations', f'augmentation_{split}.json')
    
    # 출력 디렉토리 생성
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    return {
        'dataset_dir': dataset_dir,
        'input_json': input_json,
        'input_img_dir': os.path.join(dataset_dir, split),
        'output_img_dir': output_img_dir,
        'output_json': output_json
    }

def validate_paths(paths):
    """경로 유효성 검사"""
    if not os.path.exists(paths['input_img_dir']):
        print(f"이미지 디렉토리가 존재하지 않습니다: {paths['input_img_dir']}")
        return False
        
    if not os.path.exists(paths['input_json']):
        print(f"COCO JSON 파일이 존재하지 않습니다: {paths['input_json']}")
        return False
    
    return True

def augment_split(paths, num_augmentations, visualize=False):
    """데이터셋 분할별 증강 수행"""
    print(f"\n{'='*50}")
    print(f"Processing {os.path.basename(paths['input_img_dir'])} dataset")
    print(f"{'='*50}")
    
    augment_dataset(
        json_path=paths['input_json'],
        img_dir=paths['input_img_dir'],
        output_img_dir=paths['output_img_dir'],
        output_json_path=paths['output_json'],
        num_augmentations=num_augmentations,
        visualize=visualize
    )

def main():
    # 시스템 인자로 경로 받기 또는 기본값 설정
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = '/gdrive/MyDrive/HyunsLab/Xylemrcnn'
    
    # 증강 설정
    num_augmentations = 5  # 20장을 100장 이상으로 만들기 위해 5배 증강
    splits = ['train', 'val']
    
    for split in splits:
        paths = setup_directories(base_dir, split)
        
        if not validate_paths(paths):
            print(f"Skipping {split} dataset due to missing files")
            continue
            
        # 첫 번째 split에서만 시각화
        visualize = (split == splits[0])
        augment_split(paths, num_augmentations, visualize)

if __name__ == "__main__":
    main()
