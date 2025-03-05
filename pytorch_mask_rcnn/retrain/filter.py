import json
import os
import argparse
import numpy as np

def filter_predictions(predictions_json, filtered_output_json):
    """
    예측 결과에서 원하지 않는 예측을 필터링하는 함수
    - 사용자가 수동으로 반려한 annotation ID 목록을 읽어와 해당 ID를 가진 예측 제외
    """
    # 예측 결과 로드
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)
    
    # 반려된 annotation ID 목록 로드 (예: 파일 형태나 직접 입력)
    # 여기서는 예시로 파일에서 로드하는 방식 사용
    rejected_ids_file = 'rejected_annotations.txt'
    rejected_ids = []
    
    if os.path.exists(rejected_ids_file):
        with open(rejected_ids_file, 'r') as f:
            rejected_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    
    # 반려된 annotation ID를 제외한 예측만 유지
    filtered_predictions = {"predictions": []}
    for pred in predictions["predictions"]:
        if "annotation_id" not in pred or pred["annotation_id"] not in rejected_ids:
            filtered_predictions["predictions"].append(pred)
    
    # 필터링된 예측 결과 저장
    with open(filtered_output_json, 'w') as f:
        json.dump(filtered_predictions, f, indent=2)
    
    print(f"원래 예측 수: {len(predictions['predictions'])}")
    print(f"필터링 후 예측 수: {len(filtered_predictions['predictions'])}")
    print(f"제거된 예측 수: {len(predictions['predictions']) - len(filtered_predictions['predictions'])}")
    
    return filtered_predictions

def add_to_training_set(filtered_predictions, original_annotation_file, output_annotation_file):
    """
    필터링된 예측 결과를 기존 어노테이션에 추가하여 새로운 훈련 데이터셋 생성
    """
    # 기존 어노테이션 로드
    with open(original_annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # 다음 annotation ID 계산
    next_ann_id = 1
    if annotations.get("annotations") and len(annotations["annotations"]) > 0:
        next_ann_id = max([ann["id"] for ann in annotations["annotations"]]) + 1
    
    # 다음 이미지 ID 계산
    next_img_id = 1
    if annotations.get("images") and len(annotations["images"]) > 0:
        next_img_id = max([img["id"] for img in annotations["images"]]) + 1
    
    # 필터링된 예측을 어노테이션으로 변환하여 추가
    image_ids_map = {}  # 이미지 파일명 -> 이미지 ID 매핑
    
    for pred in filtered_predictions["predictions"]:
        # 이미지 파일 확인 또는 추가
        image_file = pred.get("image_file", "unknown.png")
        
        if image_file not in image_ids_map:
            # 새 이미지 ID 할당
            image_ids_map[image_file] = next_img_id
            
            # 이미지 정보가 없으면 추가
            image_exists = False
            for img in annotations["images"]:
                if img["file_name"] == image_file:
                    image_exists = True
                    image_ids_map[image_file] = img["id"]
                    break
            
            if not image_exists:
                # 예시 이미지 정보 - 실제 구현에서는 이미지 크기 등을 정확히 추출해야 함
                annotations["images"].append({
                    "id": next_img_id,
                    "file_name": image_file,
                    "width": 800,  # 실제 이미지 크기로 대체 필요
                    "height": 600  # 실제 이미지 크기로 대체 필요
                })
                next_img_id += 1
        
        # 바운딩 박스 좌표 변환 (xyxy -> xywh 형식)
        box = pred["box"]
        if len(box) == 4:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            bbox = [x1, y1, width, height]
        else:
            bbox = box  # 이미 xywh 형식인 경우
        
        # 새 어노테이션 추가
        new_annotation = {
            "id": next_ann_id,
            "image_id": image_ids_map[image_file],
            "category_id": pred["class_id"],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "segmentation": [],  # 마스크가 있는 경우 여기에 추가
            "confidence": pred.get("score", 1.0)  # 신뢰도 정보 저장 (선택 사항)
        }
        
        # 마스크 데이터가 있는 경우 추가
        if "mask" in pred:
            # 마스크를 RLE 또는 폴리곤으로 변환하는 코드 필요
            pass
        
        annotations["annotations"].append(new_annotation)
        next_ann_id += 1
    
    # 업데이트된 어노테이션 저장
    with open(output_annotation_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"기존 어노테이션 수: {len(annotations['annotations']) - len(filtered_predictions['predictions'])}")
    print(f"새로운 어노테이션 수: {len(annotations['annotations'])}")
    print(f"추가된 어노테이션 수: {len(filtered_predictions['predictions'])}")
    
    return annotations

def main():
    parser = argparse.ArgumentParser(description="예측 필터링 및 학습 데이터셋 강화")
    parser.add_argument("--predictions", required=True, help="예측 결과 JSON 파일 경로")
    parser.add_argument("--annotations", required=True, help="원본 COCO 어노테이션 파일 경로")
    parser.add_argument("--output-annotations", required=True, help="업데이트된 어노테이션 저장 경로")
    parser.add_argument("--filtered-predictions", help="필터링된 예측 결과 저장 경로")
    
    args = parser.parse_args()
    
    # 필터링된 예측 결과 경로 설정
    if not args.filtered_predictions:
        base_name = os.path.splitext(args.predictions)[0]
        args.filtered_predictions = f"{base_name}_filtered.json"
    
    # 예측 필터링
    filtered_preds = filter_predictions(args.predictions, args.filtered_predictions)
    
    # 훈련 데이터셋에 추가
    add_to_training_set(filtered_preds, args.annotations, args.output_annotations)
    
    print(f"처리가 완료되었습니다. 업데이트된 어노테이션이 {args.output_annotations}에 저장되었습니다.")

if __name__ == "__main__":
    main()

