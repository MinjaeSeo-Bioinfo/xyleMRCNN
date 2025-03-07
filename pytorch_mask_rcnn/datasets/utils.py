from .xylem_dataset import XylemDataset

__all__ = ["datasets", "collate_wrapper"]

def datasets(ds, *args, **kwargs):
    ds = ds.lower()
    choice = ["xylem"] 
    if ds == choice[0]:
        return XylemDataset(*args, **kwargs)
    else:
        raise ValueError("'ds' must be in '{}', but got '{}'".format(choice, ds))

#---------------Batch size---------------#
def collate_wrapper(batch):
    return CustomBatch(batch)
    
class CustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.images = transposed_data[0]
        self.targets = transposed_data[1]

    def pin_memory(self):
        self.images = [img.pin_memory() for img in self.images]
        self.targets = [{k: v.pin_memory() for k, v in tgt.items()} for tgt in self.targets]
        return self
