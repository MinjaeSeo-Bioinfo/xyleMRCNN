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

def visualize_prediction(image, boxes, masks, scores, class_names, score_threshold=0.7, save_path=None):
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
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < score_threshold:
            continue
        
        # 바운딩 박스 그리기
        x1, y1, x2, y2 = box.astype(int)
        class_id = 0  # Xylem 클래스
        color = colors[class_id]
        cv2.rectangle(bbox_result, (x1, y1), (x2, y2), color, 2)
        caption = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(bbox_result, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 마스크 그리기
        if masks is not None and len(masks) > i:
            mask = masks[i]
            mask_result[mask > 0.5] = color
    
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
    
    return bbox_result, mask_result

def predict_image(model, image_path, device, score_threshold=0.7, save_dir=None):
    """
    단일 이미지에 대한 예측 수행
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지를 텐서로 변환 (0-1 범위)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 예측 수행
    with torch.no_grad():
        predictions = model(image_tensor.to(device))
    
    # 예측 결과 추출
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    masks = predictions.get('masks', None)
    
    if masks is not None:
        masks = masks.cpu().numpy()
    
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
    
    return predictions
def predict_directory(model, image_dir, device, score_threshold=0.7, save_dir=None):
    """
    디렉토리 내의 모든 이미지에 대한 예측 수행
    """
    # 이미지 파일 찾기
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
    parser.add_argument("--image-dir", required=True, help="예측할 이미지 디렉토리 경로")
    parser.add_argument("--num-classes", type=int, default=2, help="클래스 수 (배경 포함)")
    parser.add_argument("--threshold", type=float, default=0.7, help="신뢰도 임계값")
    parser.add_argument("--use-cuda", action="store_true", help="CUDA 사용 여부")
    parser.add_argument("--save-dir", help="결과를 저장할 디렉토리 경로")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드
    model = maskrcnn_resnet50(pretrained=False, num_classes=args.num_classes)
    model.to(device)
    
    # 가중치 로드
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    print("Model loaded successfully!")
    
    # 디렉토리 예측
    predict_directory(model, args.image_dir, device, args.threshold, args.save_dir)

if __name__ == "__main__":
    main()
