import copy
import torch
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class XylemEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
            
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types}
        
        self.has_results = False

    def accumulate(self, coco_results):
        # Check image IDs
        image_ids = list(set([res["image_id"] for res in coco_results]))
        gt_image_ids = list(self.coco_gt.imgs.keys())
        print(f"Result image IDs count: {len(image_ids)}")
        print(f"GT image IDs count: {len(gt_image_ids)}")
        print(f"Matching image IDs count: {len(set(image_ids).intersection(set(gt_image_ids)))}")
        
        # Check category IDs
        if len(coco_results) > 0:
            result_cats = set([res["category_id"] for res in coco_results if "category_id" in res])
            gt_cats = set(self.coco_gt.cats.keys())
            print(f"Result category IDs: {result_cats}")
            print(f"GT category IDs: {gt_cats}")
            print(f"Matching category IDs: {result_cats.intersection(gt_cats)}")
        
        # Continue with existing code
        if len(coco_results) == 0:
            return
        
        # Check loadRes function error
        try:
            image_ids = list(set([res["image_id"] for res in coco_results]))
            for iou_type in self.iou_types:
                coco_eval = self.coco_eval[iou_type]
                loaded_dt = self.coco_gt.loadRes(coco_results)
                coco_eval.cocoDt = loaded_dt
                coco_eval.params.imgIds = image_ids
                coco_eval.evaluate()
                coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
                coco_eval.accumulate()
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return
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
