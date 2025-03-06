import os
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO

# 모델 파일 import
from pytorch_mask_rcnn.model.mask_rcnn import maskrcnn_resnet50

def visualize_prediction(image, boxes, masks, scores, labels, class_names, score_threshold=0.5, save_path=None):
    """
    원본, 바운딩 박스, 마스크를 모두 시각화하는 함수
    """
    # 결과 이미지 초기화
    bbox_result = image.copy()
    mask_result = np.zeros_like(image)
    
    # 색상 생성
    colors = []
    # 배경 클래스는 랜덤 색상
    np.random.seed(0)
    colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    # Xylem 클래스는 분홍색으로 고정
    colors.append((255, 105, 180))  # 핫 핑크 색상
    
    # 예측 결과 처리
    detected_count = 0
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # 모든 결과 출력 (디버깅용)
        class_id = label  # numpy 배열이라 item() 필요 없음
        print(f"객체 {i+1}: 클래스={class_names[class_id]}, 신뢰도={score:.4f}, 위치={box}")
        
        if score < score_threshold:
            continue
        
        detected_count += 1
        
        # 바운딩 박스 그리기 부분 수정
        x1, y1, x2, y2 = box.astype(int)
        color = colors[class_id]
        # 선 두께를 더 두껍게
        cv2.rectangle(bbox_result, (x1, y1), (x2, y2), color, 4)  # 두께 증가
        
        # 텍스트 크기 및 가독성 향상
        caption = f"{class_names[class_id]}: {score:.2f}"
        text_size, _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # 텍스트 크기 증가
        cv2.rectangle(bbox_result, (x1, y1 - text_size[1] - 8), (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(bbox_result, caption, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 마스크 그리기 - 마스크 투명도 조정
        if masks is not None and len(masks) > i:
            mask = masks[i]
            # 마스크 테두리 강조를 위한 컨투어 그리기
            binary_mask = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bbox_result, contours, -1, color, 2)
            
            # 마스크 적용 (오버레이)
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask > 0.5] = color
            
            # 마스크를 반투명하게 적용 (알파 블렌딩)
            alpha = 0.5  # 투명도 조절 (0-1 사이 값)
            mask_area = (mask > 0.5).astype(np.uint8)
            bbox_result = cv2.addWeighted(bbox_result, 1, colored_mask, alpha, 0, dtype=cv2.CV_8U)
            
            # 마스크 결과 이미지에도 동일하게 적용
            mask_result[mask > 0.5] = color
    
    print(f"임계값 {score_threshold} 이상인 객체 수: {detected_count}")
    
    # 바운딩 박스와 마스크를 결합한 이미지 생성
    combined_result = bbox_result.copy()
    alpha = 0.3  # 마스크 투명도
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < score_threshold:
            continue
            
        if masks is not None and len(masks) > i:
            mask = masks[i]
            class_id = label
            color = colors[class_id]
            
            # 마스크 영역을 반투명하게 칠함
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask > 0.5] = color
            mask_area = (mask > 0.5).astype(np.uint8)
            combined_result = cv2.addWeighted(combined_result, 1, colored_mask, alpha, 0)
    
    # 결과 저장
    if save_path:
        # 4개의 이미지로 구성된 시각화
        plt.figure(figsize=(20, 10))
        
        # 원본 이미지
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image", fontsize=14)
        plt.axis('off')
        
        # 바운딩 박스 결과
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(bbox_result, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Box Predictions", fontsize=14)
        plt.axis('off')
        
        # 마스크 결과
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(mask_result, cv2.COLOR_BGR2RGB))
        plt.title("Mask Predictions", fontsize=14)
        plt.axis('off')
        
        # 결합된 결과
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(combined_result, cv2.COLOR_BGR2RGB))
        plt.title("Combined Results", fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    return bbox_result, mask_result

def predict_image(model, image_path, device, score_threshold=0.5, save_dir=None):
    """
    단일 이미지에 대한 예측 수행
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지를 텐서로 변환 (0-1 범위)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 예측 수행 - 이미지를 리스트로 감싸지 않고 직접 전달
    with torch.no_grad():
        # 모델 전달 방식을 확인
        try:
            # 먼저 직접 텐서로 시도
            predictions = model(image_tensor.to(device))
        except Exception as e1:
            print(f"직접 텐서 전달 방식 실패: {e1}")
            try:
                # 리스트로 감싸서 시도
                predictions = model([image_tensor.to(device)])
            except Exception as e2:
                print(f"리스트로 감싼 텐서 전달 방식도 실패: {e2}")
                # 배치 차원 추가 시도
                predictions = model(image_tensor.unsqueeze(0).to(device))
    
    # 예측 결과 구조 확인
    print(f"예측 결과 타입: {type(predictions)}")
    if isinstance(predictions, dict):
        print(f"예측 결과 키: {predictions.keys()}")
        if 'boxes' in predictions:
            print(f"탐지된 객체 수: {len(predictions['boxes'])}")
    elif isinstance(predictions, list):
        print(f"예측 결과 리스트 길이: {len(predictions)}")
        if len(predictions) > 0 and isinstance(predictions[0], dict):
            print(f"첫 번째 예측 결과 키: {predictions[0].keys()}")
            if 'boxes' in predictions[0]:
                print(f"탐지된 객체 수: {len(predictions[0]['boxes'])}")
    
    # 예측 결과 추출 
    if isinstance(predictions, dict):
        # 단일 결과 딕셔너리인 경우
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        masks = predictions.get('masks', None)
        if masks is not None:
            masks = masks.cpu().numpy()
    elif isinstance(predictions, list) and isinstance(predictions[0], dict):
        # 리스트 형태의 결과인 경우 (배치 처리)
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy() 
        masks = predictions[0].get('masks', None)
        if masks is not None:
            masks = masks.cpu().numpy()
    else:
        print(f"예측 결과의 형식이 예상과 다릅니다. 결과를 처리할 수 없습니다.")
        return None
    
    # 클래스 이름 목록
    class_names = ['background', 'xylem']
    
    # 결과 저장 경로 설정
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"pred_{base_name}")
    
    # 결과 시각화
    bbox_result, mask_result = visualize_prediction(image, boxes, masks, scores, 
                                               labels, class_names, score_threshold, save_path)
    
    print(f"이미지 {image_path} 예측 완료!")
    if save_path:
        print(f"결과가 {save_path}에 저장되었습니다.")
    
    return predictions

def predict_directory(model, image_dir, device, score_threshold=0.5, save_dir=None):
    """
    디렉토리 내의 모든 이미지에 대한 예측 수행
    """
    # 이미지 파일 찾기 (PNG 파일만)
    image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    
    print(f"총 {len(image_paths)}개 이미지 예측 시작")
    
    # 각 이미지에 대한 예측 수행
    results = {}
    for image_path in tqdm(image_paths):
        try:
            result = predict_image(model, image_path, device, score_threshold, save_dir)
            if result is not None:
                results[image_path] = result
        except Exception as e:
            print(f"이미지 {image_path} 예측 중 오류 발생: {e}")
    
    print(f"예측 완료. 결과가 {save_dir}에 저장되었습니다." if save_dir else "예측 완료")
    
    return results

def predict_directory_with_json(model, image_dir, device, score_threshold=0.5, save_dir=None, json_path=None):
    """
    디렉토리 내의 모든 이미지에 대한 예측 수행 후 JSON 파일 생성
    """
    # 이미지 파일 찾기
    image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    
    print(f"총 {len(image_paths)}개 이미지 예측 시작")
    
    # COCO 형식의 결과 초기화
    coco_results = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "xylem"}]
    }
    
    # 어노테이션 ID 카운터
    ann_id = 1
    
    # 각 이미지에 대한 예측 수행
    for idx, image_path in enumerate(tqdm(image_paths)):
        try:
            # 이미지 로드 및 예측
            image = cv2.imread(image_path)
            image_id = idx + 1  # 이미지 ID 생성
            
            # 이미지 정보 저장
            height, width = image.shape[:2]
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height
            }
            coco_results["images"].append(image_info)
            
            # 예측 수행
            result = predict_image(model, image_path, device, score_threshold, save_dir)
            
            if result is not None:
                # 예측 결과 가져오기 (단일 이미지 결과)
                if isinstance(result, dict):
                    boxes = result['boxes'].cpu().numpy()
                    scores = result['scores'].cpu().numpy()
                    labels = result['labels'].cpu().numpy()
                    masks = result.get('masks', None)
                    if masks is not None:
                        masks = masks.cpu().numpy()
                # 배치 처리된 결과
                elif isinstance(result, list) and isinstance(result[0], dict):
                    boxes = result[0]['boxes'].cpu().numpy()
                    scores = result[0]['scores'].cpu().numpy()
                    labels = result[0]['labels'].cpu().numpy()
                    masks = result[0].get('masks', None)
                    if masks is not None:
                        masks = masks.cpu().numpy()
                
                # 예측 결과를 COCO 형식으로 변환
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    if score < score_threshold:
                        continue
                    
                    # 바운딩 박스 (x, y, width, height) 형식으로 변환
                    x1, y1, x2, y2 = box.astype(int)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 마스크 폴리곤 생성
                    segmentation = []
                    if masks is not None and i < len(masks):
                        mask = masks[i]
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            contour = contour.flatten().tolist()
                            if len(contour) > 4:  # 최소 점 개수 확인
                                segmentation.append(contour)
                    
                    # 어노테이션 생성
                    annotation = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,  # xylem 클래스
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "area": float(width * height),
                        "segmentation": segmentation,
                        "score": float(score),
                        "iscrowd": 0
                    }
                    
                    coco_results["annotations"].append(annotation)
                    ann_id += 1
                
        except Exception as e:
            print(f"이미지 {image_path} 예측 중 오류 발생: {e}")
    
    # JSON 파일로 저장
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(coco_results, f)
        print(f"예측 결과가 {json_path}에 저장되었습니다.")
    
    print(f"예측 완료. 결과가 {save_dir}에 저장되었습니다." if save_dir else "예측 완료")
    
    return coco_results

def visualize_from_json(json_path, image_dir, output_dir):
    """
    JSON 파일의 어노테이션을 사용하여 원본 이미지에 표시
    각 bbox에 어노테이션 ID 추가
    """
    # JSON 파일 로드
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # COCO API 초기화
    coco = COCO()
    coco.dataset = coco_data
    coco.createIndex()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 이미지에 대해 시각화
    for img_info in tqdm(coco_data['images']):
        img_id = img_info['id']
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {img_path}")
            continue
        
        # 어노테이션 가져오기
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 마스크 이미지 초기화
        mask_image = np.zeros_like(image)
        
        # 각 어노테이션 처리
        for ann in anns:
            # 바운딩 박스 그리기
            bbox = ann['bbox']
            x, y, w, h = [int(v) for v in bbox]
            score = ann.get('score', 1.0)
            ann_id = ann['id']  # 어노테이션 ID 추출
            
            # 분홍색 사용 (Xylem 클래스)
            color = (255, 105, 180)
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
            
            # 레이블 텍스트 (어노테이션 ID 포함)
            caption = f"xylem (ID:{ann_id}): {score:.2f}"
            text_size, _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image, (x, y - text_size[1] - 8), (x + text_size[0] + 5, y), color, -1)
            cv2.putText(image, caption, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 마스크 그리기
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    # 세그먼트를 포인트 배열로 변환
                    if len(seg) > 4:  # 최소 점 개수 확인
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        # 마스크 채우기
                        cv2.fillPoly(mask_image, [pts], color)
                        # 마스크 테두리
                        cv2.polylines(image, [pts], True, color, 2)
        
        # 마스크 오버레이 - 개선된 방법
        alpha = 0.5
        # 마스크가 있는 위치 찾기
        mask_pixels = np.where(np.any(mask_image > 0, axis=2))
        if len(mask_pixels[0]) > 0:
            # 해당 픽셀들만 블렌딩
            for y, x in zip(mask_pixels[0], mask_pixels[1]):
                image[y, x] = image[y, x] * (1-alpha) + mask_image[y, x] * alpha
        
        # 결과 저장
        output_path = os.path.join(output_dir, f"annotated_{img_info['file_name']}")
        cv2.imwrite(output_path, image)
    
    print(f"어노테이션이 적용된 이미지가 {output_dir}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN 예측")
    parser.add_argument("--weights", required=True, help="학습된 가중치 파일 경로")
    parser.add_argument("--input", required=True, help="예측할 이미지 경로 또는 디렉토리 경로")
    parser.add_argument("--num-classes", type=int, default=2, help="클래스 수 (배경 포함)")
    parser.add_argument("--threshold", type=float, default=0.5, help="신뢰도 임계값")
    parser.add_argument("--use-cuda", action="store_true", help="CUDA 사용 여부")
    parser.add_argument("--save-dir", help="결과를 저장할 디렉토리 경로")
    parser.add_argument("--json-path", help="예측 결과를 저장할 JSON 파일 경로")
    parser.add_argument("--visualize", action="store_true", help="JSON에서 어노테이션 시각화")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드
    try:
        # weights=None
        model = maskrcnn_resnet50(weights=None, num_classes=args.num_classes)
    except TypeError:
        # 실패하면 pretrained=False
        model = maskrcnn_resnet50(pretrained=False, num_classes=args.num_classes)
    model.to(device)
    
    # 가중치 로드
    print(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("Model loaded successfully!")
    
    # 입력이 디렉토리인지 파일인지 확인
    is_directory = os.path.isdir(args.input)
    
    if is_directory:
        # JSON 파일 생성
        if args.json_path:
            predict_directory_with_json(model, args.input, device, args.threshold, args.save_dir, args.json_path)
            
            # JSON 파일로부터 어노테이션 시각화
            if args.visualize and args.json_path:
                output_dir = os.path.join(args.save_dir, "annotated") if args.save_dir else "annotated_images"
                visualize_from_json(args.json_path, args.input, output_dir)
        else:
            predict_directory(model, args.input, device, args.threshold, args.save_dir)
    else:
        # 단일 이미지 예측
        if not args.input.lower().endswith('.png'):
            print("경고: 입력 파일이 PNG 형식이 아닙니다. 처리가 실패할 수 있습니다.")
        
        predict_image(model, args.input, device, args.threshold, args.save_dir)

if __name__ == "__main__":
    main()

