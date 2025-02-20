import copy
import torch
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

class XylemEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
            
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        #self.ann_labels = ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types}
        
        self.has_results = False
    
    # input all predictions
    def accumulate(self, coco_results):
        if len(coco_results) == 0:
            return
        
        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = self.coco_gt.loadRes(coco_results)
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval.accumulate()
            
        self.has_results = True
    
    def summarize(self):
        if self.has_results:
            for iou_type in self.iou_types:
                print(f"IoU metric: {iou_type}")
                self.coco_eval[iou_type].summarize()
        else:
            print("evaluation has no results")

def prepare_for_xylem_coco(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks"]
        
        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
        boxes = boxes.tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        
        #@ Debugging
        if len(labels) > 0:
            print(f"Original labels (first 5): {labels[:5]}")
        
        # to fit category xylem
        labels = [label - 1 for label in labels]
        
        masks = masks > 0.5
        rles = [
            mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
            
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[i],
                    "bbox": boxes[i],
                    "segmentation": rle,
                    "score": scores[i],
                }
                for i, rle in enumerate(rles)
            ]
        )

    return coco_results
    
    #     #@@@@ result가 빈 경우 포함 x @@@@@@@@@
    #     results = [
    #         {
    #             "image_id": original_id,
    #             "category_id": labels[i],
    #             "bbox": boxes[i],
    #             "segmentation": rle,
    #             "score": scores[i],
    #         }
    #         for i, rle in enumerate(rles)
    #     ]
        
    #     if len(results) > 0:
    #         print(f"Generated {len(results)} results for image {original_id}")
    #         print(f"Sample category_id: {results[0]['category_id']}")
            
    #     coco_results.extend(results)
    # print(f"Total results: {len(coco_results)}")
    #     #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
