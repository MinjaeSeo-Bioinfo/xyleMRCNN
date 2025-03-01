import os
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

# 모델 파일 import
from pytorch_mask_rcnn.model.mask_rcnn import maskrcnn_resnet50

def visualize_prediction(image, boxes, masks, scores, class_names, score_threshold=0.5, save_path=None):
    """
    원본, 바운딩 박스, 마스크를 모두 시각화하는 함수
    """
    # 결과 이미지 초기화
    bbox_result = image.copy()
    mask_result = np.zeros_like(image)
    
    # 색상 생성 (클래스별로 다른 색상)
    num_classes = len(class_names)
    colors = []
    for i in range(num_classes):
        np.random.seed(i * 10)
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    
    # 예측 결과 처리
    detected_count = 0
    for i, (box, score) in enumerate(zip(boxes, scores)):
        # 모든 결과 출력 (디버깅용)
        class_id = 0  # Xylem 클래스
        print(f"객체 {i+1}: 클래스={class_names[class_id]}, 신뢰도={score:.4f}, 위치={box}")
        
        if score < score_threshold:
            continue
        
        detected_count += 1
        
        # 바운딩 박스 그리기
        x1, y1, x2, y2 = box.astype(int)
        color = colors[class_id]
        cv2.rectangle(bbox_result, (x1, y1), (x2, y2), color, 2)
        caption = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(bbox_result, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 마스크 그리기
        if masks is not None and len(masks) > i:
            mask = masks[i]
            mask_result[mask > 0.5] = color
    
    print(f"임계값 {score_threshold} 이상인 객체 수: {detected_count}")
    
    # 결과 저장 또는 표시
    if save_path:
        plt.figure(figsize=(18, 6))
        
        # 원본 이미지
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # 바운딩 박스 결과
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(bbox_result, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Box Predictions")
        plt.axis('off')
        
        # 마스크 결과
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(mask_result, cv2.COLOR_BGR2RGB))
        plt.title("Mask Predictions")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # 추가로 임계값을 낮춘 결과도 저장
        if detected_count == 0:
            print("기본 임계값으로 검출된 객체가 없습니다. 낮은 임계값으로 다시 시도합니다.")
            lower_threshold = 0.1
            base_name = os.path.splitext(save_path)[0]
            lower_save_path = f"{base_name}_lower_threshold{lower_threshold}.png"
            
            # 임계값을 낮춘 결과 생성
            lower_bbox_result = image.copy()
            lower_mask_result = np.zeros_like(image)
            
            lower_detected_count = 0
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score < lower_threshold:
                    continue
                
                lower_detected_count += 1
                
                # 바운딩 박스 그리기
                x1, y1, x2, y2 = box.astype(int)
                class_id = 0  # Xylem 클래스
                color = colors[class_id]
                cv2.rectangle(lower_bbox_result, (x1, y1), (x2, y2), color, 2)
                caption = f"{class_names[class_id]}: {score:.2f}"
                cv2.putText(lower_bbox_result, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 마스크 그리기
                if masks is not None and len(masks) > i:
                    mask = masks[i]
                    lower_mask_result[mask > 0.5] = color
            
            plt.figure(figsize=(18, 6))
            
            # 원본 이미지
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')
            
            # 바운딩 박스 결과
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(lower_bbox_result, cv2.COLOR_BGR2RGB))
            plt.title(f"Bounding Box (threshold={lower_threshold})")
            plt.axis('off')
            
            # 마스크 결과
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(lower_mask_result, cv2.COLOR_BGR2RGB))
            plt.title(f"Mask (threshold={lower_threshold})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(lower_save_path)
            plt.close()
            
            print(f"임계값 {lower_threshold}으로 {lower_detected_count}개 객체 검출, 결과가 {lower_save_path}에 저장되었습니다.")
    
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
    
    # 예측 결과 추출 (데이터 구조에 따라 적절히 처리)
    if isinstance(predictions, dict):
        # 단일 결과 딕셔너리인 경우
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions.get('masks', None)
        if masks is not None:
            masks = masks.cpu().numpy()
    elif isinstance(predictions, list) and isinstance(predictions[0], dict):
        # 리스트 형태의 결과인 경우 (배치 처리)
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
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
                                                  class_names, score_threshold, save_path)
    
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

def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN 예측")
    parser.add_argument("--weights", required=True, help="학습된 가중치 파일 경로")
    parser.add_argument("--input", required=True, help="예측할 이미지 경로 또는 디렉토리 경로")
    parser.add_argument("--num-classes", type=int, default=2, help="클래스 수 (배경 포함)")
    parser.add_argument("--threshold", type=float, default=0.5, help="신뢰도 임계값")
    parser.add_argument("--use-cuda", action="store_true", help="CUDA 사용 여부")
    parser.add_argument("--save-dir", help="결과를 저장할 디렉토리 경로")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드 - 가중치 다운로드 방지
    try:
        # weights=None으로 명시적으로 설정하여 ImageNet 가중치 다운로드 방지
        model = maskrcnn_resnet50(weights=None, num_classes=args.num_classes)
    except TypeError:
        # 구버전 torchvision에서는 weights 인자 대신 pretrained 사용
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
        # 디렉토리 예측
        predict_directory(model, args.input, device, args.threshold, args.save_dir)
    else:
        # 단일 이미지 예측
        if not args.input.lower().endswith('.png'):
            print("경고: 입력 파일이 PNG 형식이 아닙니다. 처리가 실패할 수 있습니다.")
        
        predict_image(model, args.input, device, args.threshold, args.save_dir)

if __name__ == "__main__":
    main()
