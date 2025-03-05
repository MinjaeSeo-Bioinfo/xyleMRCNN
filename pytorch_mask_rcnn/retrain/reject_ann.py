import json
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

class AnnotationSelectorColab:
    def __init__(self, image_path, predictions_json):
        # 이미지 및 예측 로드
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        with open(predictions_json, 'r') as f:
            self.predictions = json.load(f)
        
        # 상태 관리
        self.rejected_ids = set()
        self.current_pred_idx = 0
        self.total_preds = len(self.predictions['predictions'])
        
        # 위젯 초기화
        self.setup_widgets()
        
    def setup_widgets(self):
        # 버튼 생성
        self.reject_button = widgets.Button(
            description='Reject',
            button_style='danger',
            tooltip='이 예측을 반려합니다'
        )
        
        self.accept_button = widgets.Button(
            description='Accept',
            button_style='success',
            tooltip='이 예측을 수락합니다'
        )
        
        self.prev_button = widgets.Button(
            description='Previous',
            tooltip='이전 예측으로 이동'
        )
        
        self.next_button = widgets.Button(
            description='Next',
            tooltip='다음 예측으로 이동'
        )
        
        self.save_button = widgets.Button(
            description='Save & Exit',
            button_style='info',
            tooltip='반려된 ID를 저장하고 종료'
        )
        
        # 진행 상황 표시기
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=self.total_preds-1,
            description='Progress:',
            bar_style='info'
        )
        
        # 상태 레이블
        self.status_label = widgets.Label(value='준비 완료. 예측을 검토하세요.')
        
        # 이벤트 연결
        self.reject_button.on_click(self.reject)
        self.accept_button.on_click(self.accept)
        self.prev_button.on_click(self.prev)
        self.next_button.on_click(self.next)
        self.save_button.on_click(self.save)
        
        # 위젯 레이아웃
        self.nav_buttons = widgets.HBox([self.prev_button, self.next_button])
        self.action_buttons = widgets.HBox([self.reject_button, self.accept_button, self.save_button])
        self.controls = widgets.VBox([
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
        
        # 초기 예측 표시
        self.show_current_prediction()
        
    def show_current_prediction(self):
        with self.output:
            clear_output(wait=True)
            
            # 이미지 복사
            img_copy = self.image.copy()
            
            # 현재 예측 정보
            if self.current_pred_idx < self.total_preds:
                pred = self.predictions['predictions'][self.current_pred_idx]
                
                # 바운딩 박스 추출
                box = pred['box']
                if len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                else:
                    print("박스 형식이 맞지 않습니다.")
                    return
                
                # 바운딩 박스 그리기
                color = (255, 0, 0) if pred.get('annotation_id') in self.rejected_ids else (0, 255, 0)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                
                # 텍스트 정보
                status = "REJECTED" if pred.get('annotation_id') in self.rejected_ids else "ACCEPTED"
                class_name = pred.get('class_name', 'Unknown')
                score = pred.get('score', 0.0)
                ann_id = pred.get('annotation_id', 'N/A')
                
                label = f"{class_name}: {score:.2f}, ID: {ann_id}, Status: {status}"
                cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 진행 상황 업데이트
                self.progress.value = self.current_pred_idx
                
                # 이미지 표시
                plt.figure(figsize=(12, 8))
                plt.title(f"예측 {self.current_pred_idx + 1}/{self.total_preds}: {label}")
                plt.imshow(img_copy)
                plt.axis('off')
                plt.show()
            else:
                plt.figure(figsize=(8, 6))
                plt.title("모든 예측을 검토했습니다.")
                plt.text(0.5, 0.5, "검토 완료", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20)
                plt.axis('off')
                plt.show()
    
    def reject(self, b):
        if self.current_pred_idx < self.total_preds:
            pred = self.predictions['predictions'][self.current_pred_idx]
            ann_id = pred.get('annotation_id')
            if ann_id is not None:
                if ann_id in self.rejected_ids:
                    self.rejected_ids.remove(ann_id)
                    self.status_label.value = f"ID {ann_id}를 반려 목록에서 제거했습니다."
                else:
                    self.rejected_ids.add(ann_id)
                    self.status_label.value = f"ID {ann_id}를 반려 목록에 추가했습니다."
                self.show_current_prediction()
                self.next(None)  # 자동으로 다음 예측으로 이동
    
    def accept(self, b):
        if self.current_pred_idx < self.total_preds:
            pred = self.predictions['predictions'][self.current_pred_idx]
            ann_id = pred.get('annotation_id')
            if ann_id is not None and ann_id in self.rejected_ids:
                self.rejected_ids.remove(ann_id)
                self.status_label.value = f"ID {ann_id}를 수락했습니다."
                self.show_current_prediction()
            self.next(None)  # 자동으로 다음 예측으로 이동
    
    def prev(self, b):
        if self.current_pred_idx > 0:
            self.current_pred_idx -= 1
            self.show_current_prediction()
    
    def next(self, b):
        if self.current_pred_idx < self.total_preds - 1:
            self.current_pred_idx += 1
            self.show_current_prediction()
    
    def save(self, b):
        # 반려된 annotation ID 저장
        with open('rejected_annotations.txt', 'w') as f:
            for ann_id in sorted(self.rejected_ids):
                f.write(f"{ann_id}\n")
        
        self.status_label.value = f"반려된 {len(self.rejected_ids)}개의 annotation ID가 저장되었습니다."
        with self.output:
            clear_output(wait=True)
            print(f"반려된 {len(self.rejected_ids)}개의 annotation ID가 저장되었습니다.")
            print("반려된 ID 목록:")
            print(sorted(self.rejected_ids))

def start_annotation_selector(image_path, predictions_json):
    """Colab에서 실행할 수 있는 주 함수"""
    # 파일 존재 여부 확인
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    if not os.path.exists(predictions_json):
        print(f"예측 결과 파일을 찾을 수 없습니다: {predictions_json}")
        return
    
    # 어노테이션 선택기 시작
    selector = AnnotationSelectorColab(image_path, predictions_json)
    return selector

# Colab에서 실행하기 위한 예제
"""
from colab_reject_annotations import start_annotation_selector

# 사용 예시
image_path = '/content/sample_image.png'  # 이미지 경로
predictions_json = '/content/predictions.json'  # 예측 결과 JSON 파일 경로

selector = start_annotation_selector(image_path, predictions_json)
"""

