import os
import re
import random
import torch


__all__ = ["save_ckpt", "Meter"]

def save_ckpt(model, optimizer, epochs, ckpt_path, **kwargs):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]  = optimizer.state_dict()
    checkpoint["epochs"] = epochs
        
    for k, v in kwargs.items():
        checkpoint[k] = v
        
    prefix, ext = os.path.splitext(ckpt_path)
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, ckpt_path)

class TextArea:
    def __init__(self):
        self.buffer = []

    def write(self, s):
        self.buffer.append(s)

    def __str__(self):
        return "".join(self.buffer)

    def extract_ap(self, lines):
        """Extract AP values from filtered lines"""
        for line in lines:
            if "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line:
                match = re.search(r"(-?[0-9]+\.[0-9]+)", line)
                if match:
                    ap_value = float(match.group(1))
                    print(f"Extracted AP from line '{line}': {ap_value}")  # 디버깅 추가
                    return max(ap_value, 0.0)  # -1.000이면 0.0 반환
        return 0.0 

    def get_AP(self):
        result = {"bbox AP": 0.0, "mask AP": 0.0}

        txt = str(self)
        print("Evaluation Output:", txt)
        
        # Separate bbox & segm sections
        bbox_section = txt.split("IoU metric: bbox")[-1].split("IoU metric: segm")[0]
        mask_section = txt.split("IoU metric: segm")[-1]
        
        # Filter AP values
        bbox_ap_lines = [line for line in bbox_section.split("\n") if "Average Precision" in line]
        mask_ap_lines = [line for line in mask_section.split("\n") if "Average Precision" in line]

        print("Filtered BBox AP Lines:", bbox_ap_lines)  # 디버깅 추가
        print("Filtered Mask AP Lines:", mask_ap_lines)  # 디버깅 추가

        try:
            result["bbox AP"] = self.extract_ap(bbox_ap_lines)
            result["mask AP"] = self.extract_ap(mask_ap_lines)
        except Exception as e:
            print("Failed to parse AP values:", e)

        print("Final AP result:", result)  # 최종 결과 확인
        return result

class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)
    
                
# class TextArea:
#     def __init__(self):
#         self.buffer = []
    
#     def write(self, s):
#         self.buffer.append(s)
        
#     def __str__(self):
#         return "".join(self.buffer)

#     def get_AP(self):
#         result = {"bbox AP": 0.0, "mask AP": 0.0}
        
#         txt = str(self)
#         values = re.findall(r"(\d{3})\n", txt)
#         if len(values) > 0:
#             values = [int(v) / 10 for v in values]
#             result = {"bbox AP": values[0], "mask AP": values[12]}
            
#         return result
