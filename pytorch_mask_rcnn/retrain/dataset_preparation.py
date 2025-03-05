import os
import glob
import json
import cv2
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import shutil

class PredictionReviewer:
    def __init__(self, images_dir, predictions_json, filename_prefix="annotated_", output_dir=None):
        """
        예측 결과를 검토하고 필터링하는 클래스
        
        Args:
            images_dir: 이미지 파일이 있는 디렉토리
            predictions_json: 예측 결과가 포함된 JSON 파일 경로
            filename_prefix: 이미지 파일 이름 접두사 (기본값: "annotated_")
            output_dir: 출력 파일을 저장할 디렉토리 (기본값: 현재 디렉토리)
        """
        self.images_dir = images_dir
        self.predictions_json = predictions_json
        self.filename_prefix = filename_prefix
        self.output_dir = output_dir or os.getcwd()  # 기본값은 현재 디렉토리
        
        # 예측 JSON 로드
        with open(predictions_json, 'r') as f:
            self.predictions_data = json.load(f)
        
        # 이미지 정보 매핑
        self.image_info = {}
        for img in self.predictions_data.get('images', []):
            self.image_info[img['id']] = img
        
        # 어노테이션 정보 매핑 (이미지 ID별)
        self.image_annotations = {}
        for ann in self.predictions_data.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # 실제 파일 이름 매핑 구성
        self.filename_mapping = {}
        self.reverse_mapping = {}
        for img_id, img_data in self.image_info.items():
            original_name = img_data['file_name']
            prefixed_name = f"{filename_prefix}{original_name}"
            self.filename_mapping[original_name] = prefixed_name
            self.reverse_mapping[prefixed_name] = original_name
        
        # 모든 이미지 ID 목록
        self.image_ids = sorted(list(self.image_info.keys()))
        
        # 반려된 annotation ID 집합
        self.rejected_ids = set()
        
        # 현재 상태
        self.current_img_idx = 0
        self.current_ann_idx = 0
        
        print(f"총 {len(self.image_ids)}개 이미지의 정보를 로드했습니다.")
        print(f"총 {sum(len(anns) for anns in self.image_annotations.values())}개의 어노테이션이 있습니다.")
        print(f"출력 파일은 {self.output_dir}에 저장됩니다.")
        
        # UI 초기화
        self.setup_widgets()
    
    def setup_widgets(self):
        # 이미지 선택 드롭다운 (실제 파일 이름으로 표시)
        img_options = [(f"{i}: {self.image_info[img_id]['file_name']}", img_id) 
                       for i, img_id in enumerate(self.image_ids)]
        
        self.image_dropdown = widgets.Dropdown(
            options=img_options,
            description='이미지:',
            style={'description_width': 'initial'}
        )
        self.image_dropdown.observe(self.on_image_change, names='value')
        
        # 버튼 생성
        self.reject_button = widgets.Button(
            description='Reject',
            button_style='danger',
            tooltip='이 어노테이션을 반려합니다'
        )
        
        self.accept_button = widgets.Button(
            description='Accept',
            button_style='success',
            tooltip='이 어노테이션을 수락합니다'
        )
        
        self.prev_button = widgets.Button(
            description='Previous',
            tooltip='이전 어노테이션으로 이동'
        )
        
        self.next_button = widgets.Button(
            description='Next',
            tooltip='다음 어노테이션으로 이동'
        )
        
        self.save_button = widgets.Button(
            description='Save Filtered Predictions',
            button_style='info',
            tooltip='필터링된 예측을 저장합니다'
        )
        
        # 진행 상황 표시기
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=1,  # 나중에 업데이트됨
            description='Progress:',
            bar_style='info'
        )
        
        # 상태 레이블
        self.status_label = widgets.Label(value='준비 완료. 이미지를 선택하세요.')
        
        # 이벤트 연결
        self.reject_button.on_click(self.reject)
        self.accept_button.on_click(self.accept)
        self.prev_button.on_click(self.prev)
        self.next_button.on_click(self.next)
        self.save_button.on_click(self.save_filtered_predictions)
        
        # 위젯 레이아웃
        self.nav_buttons = widgets.HBox([self.prev_button, self.next_button])
        self.action_buttons = widgets.HBox([self.reject_button, self.accept_button, self.save_button])
        self.controls = widgets.VBox([
            self.image_dropdown,
            self.progress,
            self.nav_buttons,
            self.action_buttons,
            self.status_label
        ])
        
        # 출력 영역
        self.output = widgets.Output()
        
        # 위젯 표시
        display(self.controls)
        display(self.output)
    
    def on_image_change(self, change):
        """이미지 드롭다운 변경 시 호출되는 함수"""
        if change['type'] == 'change' and change['name'] == 'value':
            self.current_img_id = change['new']
            self.current_ann_idx = 0
            self.load_current_image()
            self.show_current_annotation()
    
    def load_current_image(self):
        """현재 선택된 이미지 로드"""
        if not self.current_img_id:
            return
        
        img_data = self.image_info[self.current_img_id]
        original_filename = img_data['file_name']
        prefixed_filename = self.filename_mapping[original_filename]
        
        image_path = os.path.join(self.images_dir, prefixed_filename)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(image_path):
            # 접두사 없는 파일명도 시도
            image_path = os.path.join(self.images_dir, original_filename)
            if not os.path.exists(image_path):
                self.status_label.value = f"이미지 파일을 찾을 수 없습니다: {prefixed_filename} 또는 {original_filename}"
                self.image = None
                return
        
        self.image = cv2.imread(image_path)
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            self.status_label.value = f"이미지를 로드할 수 없습니다: {image_path}"
            self.image = None
            return
        
        # 현재 이미지의 어노테이션 수 업데이트
        current_anns = self.image_annotations.get(self.current_img_id, [])
        self.progress.max = len(current_anns) - 1 if len(current_anns) > 0 else 0
        
        self.status_label.value = f"이미지 로드 완료: {os.path.basename(image_path)}"
    
    def show_current_annotation(self):
        """현재 어노테이션 표시"""
        with self.output:
            clear_output(wait=True)
            
            if self.image is None:
                print("이미지를 로드할 수 없습니다.")
                return
            
            # 현재 이미지의 어노테이션 목록
            current_anns = self.image_annotations.get(self.current_img_id, [])
            
            if not current_anns:
                print(f"이미지 ID {self.current_img_id}에 대한 어노테이션이 없습니다.")
                plt.figure(figsize=(12, 8))
                plt.imshow(self.image)
                plt.title("어노테이션 없음")
                plt.axis('off')
                plt.show()
                return
            
            # 이미지 복사
            img_copy = self.image.copy()
            
            # 현재 어노테이션
            if self.current_ann_idx < len(current_anns):
                ann = current_anns[self.current_ann_idx]
                
                # 바운딩 박스 추출 (COCO 형식 [x, y, width, height]을 [x1, y1, x2, y2]로 변환)
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = map(float, bbox)
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    
                    # 바운딩 박스 그리기
                    ann_id = ann.get('id')
                    color = (255, 0, 0) if ann_id in self.rejected_ids else (0, 255, 0)
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    
                    # 텍스트 정보
                    status = "REJECTED" if ann_id in self.rejected_ids else "ACCEPTED"
                    class_id = ann.get('category_id', 0)
                    score = ann.get('score', 1.0)
                    
                    label = f"Class: {class_id}, ID: {ann_id}, Score: {score:.2f}, Status: {status}"
                    cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 진행 상황 업데이트
                    self.progress.value = self.current_ann_idx
                    
                    # 이미지 표시 (2x2 레이아웃)
                    plt.figure(figsize=(15, 10))
                    
                    # 전체 이미지
                    plt.subplot(2, 2, 1)
                    img_filename = self.image_info[self.current_img_id]['file_name']
                    plt.title(f"{img_filename} - 어노테이션 {self.current_ann_idx + 1}/{len(current_anns)}")
                    plt.imshow(img_copy)
                    plt.axis('off')
                    
                    # 확대된 바운딩 박스 영역
                    plt.subplot(2, 2, 2)
                    plt.title(f"확대 보기: {label}")
                    
                    # 패딩 추가 (바운딩 박스 주변 영역 포함)
                    padding = max(30, int(max(w, h) * 0.3))
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(img_copy.shape[1], x2 + padding)
                    crop_y2 = min(img_copy.shape[0], y2 + padding)
                    
                    # 확대 영역 자르기
                    zoomed = img_copy[crop_y1:crop_y2, crop_x1:crop_x2]
                    plt.imshow(zoomed)
                    plt.axis('off')
                    
                    # 원본 이미지와 다른 모든 어노테이션 표시
                    full_img_with_all = self.image.copy()
                    
                    # 모든 어노테이션 그리기
                    for i, other_ann in enumerate(current_anns):
                        other_bbox = other_ann.get('bbox', [])
                        if len(other_bbox) == 4:
                            other_x, other_y, other_w, other_h = map(float, other_bbox)
                            other_x1, other_y1, other_x2, other_y2 = int(other_x), int(other_y), int(other_x + other_w), int(other_y + other_h)
                            
                            # 현재 어노테이션 확인
                            other_ann_id = other_ann.get('id')
                            is_current = (i == self.current_ann_idx)
                            is_rejected = other_ann_id in self.rejected_ids
                            
                            # 색상 설정: 현재 = 빨강/초록, 다른 것 = 파랑
                            if is_current:
                                box_color = (255, 0, 0) if is_rejected else (0, 255, 0)
                                thickness = 3
                            else:
                                box_color = (0, 0, 255) if is_rejected else (100, 100, 255)
                                thickness = 1
                                
                            cv2.rectangle(full_img_with_all, (other_x1, other_y1), (other_x2, other_y2), box_color, thickness)
                    
                    plt.subplot(2, 2, 3)
                    plt.title("모든 어노테이션")
                    plt.imshow(full_img_with_all)
                    plt.axis('off')
                    
                    # 현재 어노테이션이 있는 확대된 영역 표시
                    full_zoomed = self.image.copy()
                    rect_color = (255, 255, 0)  # 노란색 사각형으로 현재 확대 영역 표시
                    cv2.rectangle(full_zoomed, (crop_x1, crop_y1), (crop_x2, crop_y2), rect_color, 2)
                    
                    plt.subplot(2, 2, 4)
                    plt.title("확대 영역")
                    plt.imshow(full_zoomed)
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"바운딩 박스 형식이 잘못되었습니다: {bbox}")
            else:
                print(f"유효한 어노테이션 인덱스가 아닙니다: {self.current_ann_idx}")
    
    def reject(self, b):
        """현재 어노테이션 반려"""
        current_anns = self.image_annotations.get(self.current_img_id, [])
        
        if self.current_ann_idx < len(current_anns):
            ann = current_anns[self.current_ann_idx]
            ann_id = ann.get('id')
            
            if ann_id is not None:
                if ann_id in self.rejected_ids:
                    self.rejected_ids.remove(ann_id)
                    self.status_label.value = f"ID {ann_id}를 반려 목록에서 제거했습니다."
                else:
                    self.rejected_ids.add(ann_id)
                    self.status_label.value = f"ID {ann_id}를 반려 목록에 추가했습니다."
                
                self.show_current_annotation()
                self.next(None)  # 자동으로 다음 어노테이션으로 이동
    
    def accept(self, b):
        """현재 어노테이션 수락"""
        current_anns = self.image_annotations.get(self.current_img_id, [])
        
        if self.current_ann_idx < len(current_anns):
            ann = current_anns[self.current_ann_idx]
            ann_id = ann.get('id')
            
            if ann_id is not None and ann_id in self.rejected_ids:
                self.rejected_ids.remove(ann_id)
                self.status_label.value = f"ID {ann_id}를 수락했습니다."
                self.show_current_annotation()
            
            self.next(None)  # 자동으로 다음 어노테이션으로 이동
    
    def prev(self, b):
        """이전 어노테이션으로 이동"""
        if self.current_ann_idx > 0:
            self.current_ann_idx -= 1
            self.show_current_annotation()
    
    def next(self, b):
        """다음 어노테이션으로 이동"""
        current_anns = self.image_annotations.get(self.current_img_id, [])
        
        if self.current_ann_idx < len(current_anns) - 1:
            self.current_ann_idx += 1
            self.show_current_annotation()
    
    def save_filtered_predictions(self, b):
        """필터링된 예측 저장"""
        # 반려된 ID 목록 저장 (나중에 참조용)
        rejected_path = os.path.join(self.output_dir, 'rejected_annotations.txt')
        with open(rejected_path, 'w') as f:
            for ann_id in sorted(self.rejected_ids):
                f.write(f"{ann_id}\n")
        
        # 필터링된 어노테이션 목록 생성 (반려된 ID 제외)
        filtered_annotations = []
        for annotations in self.image_annotations.values():
            for ann in annotations:
                if ann.get('id') not in self.rejected_ids:
                    filtered_annotations.append(ann)
        
        # 어노테이션 ID 재할당 (연속적인 번호로)
        for i, ann in enumerate(filtered_annotations, 1):
            ann['id'] = i
        
        # 필터링된 예측 데이터 생성
        filtered_predictions = {
            'images': list(self.image_info.values()),
            'annotations': filtered_annotations,
            'categories': self.predictions_data.get('categories', [])
        }
        
        # 이미지 ID도 연속적인 번호로 재할당
        for i, img in enumerate(filtered_predictions['images'], 1):
            old_id = img['id']
            img['id'] = i
            
            # 해당 이미지의 어노테이션도 image_id 업데이트
            for ann in filtered_annotations:
                if ann['image_id'] == old_id:
                    ann['image_id'] = i
        
        # 필터링된 예측 저장
        output_json = os.path.join(self.output_dir, 'filtered_predictions.json')
        with open(output_json, 'w') as f:
            json.dump(filtered_predictions, f, indent=2)
        
        # 이미지별 반려된 ID 정보 저장 (디버깅용)
        rejected_ids_json = os.path.join(self.output_dir, 'image_to_rejected_ids.json')
        image_to_rejected = {}
        for img_id in self.image_ids:
            img_filename = self.image_info[img_id]['file_name']
            anns = self.image_annotations.get(img_id, [])
            rejected_in_img = []
            
            for ann in anns:
                ann_id = ann.get('id')
                if ann_id in self.rejected_ids:
                    rejected_in_img.append(ann_id)
            
            if rejected_in_img:
                image_to_rejected[img_filename] = rejected_in_img
        
        with open(rejected_ids_json, 'w') as f:
            json.dump(image_to_rejected, f, indent=2)
        
        # 훈련용 데이터셋 준비
        try:
            train_dir = self.prepare_training_dataset()
            success_msg = f"재학습용 데이터셋이 {train_dir} 디렉토리에 준비되었습니다."
        except Exception as e:
            success_msg = f"훈련용 데이터셋 준비 중 오류 발생: {str(e)}"
        
        self.status_label.value = f"필터링된 예측이 {output_json}에 저장되었습니다. {success_msg}"
        with self.output:
            clear_output(wait=True)
            print(f"총 {len(self.rejected_ids)}개의 어노테이션이 반려되었습니다.")
            print(f"필터링된 어노테이션 수: {len(filtered_annotations)}")
            print(f"필터링된 예측이 {output_json}에 저장되었습니다.")
            print(success_msg)
    
    def prepare_training_dataset(self):
        """재학습을 위한 데이터셋 준비 (필터링된 예측 + 이미지)"""
        # 훈련용 디렉토리 생성
        train_dir = os.path.join(self.output_dir, "retrain_dataset")
        os.makedirs(train_dir, exist_ok=True)
        
        # 어노테이션 디렉토리 생성
        ann_dir = os.path.join(train_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        
        # 이미지 디렉토리 생성
        img_dir = os.path.join(train_dir, "train")
        os.makedirs(img_dir, exist_ok=True)
        
        # 필터링된 예측을 어노테이션 파일로 저장
        filtered_json_path = os.path.join(ann_dir, "result_train.json")
        filtered_predictions_path = os.path.join(self.output_dir, "filtered_predictions.json")
        shutil.copy(filtered_predictions_path, filtered_json_path)
        
        # 이미지 복사
        for img_id, img_info in self.image_info.items():
            original_name = img_info['file_name']
            prefixed_name = self.filename_mapping.get(original_name, original_name)
            
            # 이미지 파일 경로
            src_path = os.path.join(self.images_dir, prefixed_name)
            if not os.path.exists(src_path):
                src_path = os.path.join(self.images_dir, original_name)
            
            if os.path.exists(src_path):
                # 훈련용 디렉토리에 복사 (접두사 제거)
                dst_path = os.path.join(img_dir, original_name)
                shutil.copy(src_path, dst_path)
            else:
                print(f"이미지를 찾을 수 없습니다: {prefixed_name} 또는 {original_name}")
        
        print(f"재학습용 데이터셋이 {train_dir} 디렉토리에 준비되었습니다.")
        print(f"이미지 수: {len(self.image_info)}")
        
        return train_dir

# Colab에서 사용 예시
def start_prediction_review(images_dir, predictions_json, filename_prefix="annotated_", output_dir=None):
    """
    예측 검토 시작
    
    Args:
        images_dir: 이미지 파일이 있는 디렉토리
        predictions_json: 예측 결과가 포함된 JSON 파일 경로
        filename_prefix: 이미지 파일 이름 접두사 (기본값: "annotated_")
        output_dir: 출력 파일을 저장할 디렉토리 (기본값: 현재 디렉토리)
    
    Returns:
        reviewer: PredictionReviewer 객체
    """
    if not os.path.exists(images_dir):
        print(f"이미지 디렉토리를 찾을 수 없습니다: {images_dir}")
        return
    
    if not os.path.exists(predictions_json):
        print(f"예측 결과 파일을 찾을 수 없습니다: {predictions_json}")
        return
    
    # 출력 디렉토리가 없으면 생성
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    reviewer = PredictionReviewer(images_dir, predictions_json, filename_prefix, output_dir)
    return reviewer

