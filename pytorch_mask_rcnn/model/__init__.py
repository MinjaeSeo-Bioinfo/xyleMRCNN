try:
    from .mask_rcnn import maskrcnn_resnet50
except ImportError:
    from .mask_rcnn import maskrcnn_se_resnet50 as maskrcnn_resnet50
