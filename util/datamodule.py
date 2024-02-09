from datasets import Toybox5
import torch
from torch.utils.data import DataLoader

class Toybox5DataModule():
    def __init__(self, 
                 scene_list: list = [],
                 root: str = '/mnt/data/toybox/toybox-5',
                 batch_size: int = 64, 
                 num_workers: int = 4,):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root    
        self.scene_list = scene_list

    def setup(self, stage=None):
        train_dataset_list = []
        # test_dataset_list = []

        for scene in self.scene_list:
            train_dataset_list.append(Toybox5(scene_root=self.root, split='train', scene=scene))
            # test_dataset_list.append(Toybox5(scene_root=self.root, split='test', scene=scene))
        
        self.toybox5_train = torch.utils.data.ConcatDataset(train_dataset_list)
        # self.toybox5_test = torch.utils.data.ConcatDataset(test_dataset_list)

    def train_dataloader(self):
        return self.toybox5_train

    # def test_dataloader(self):
    #     return self.toybox5_test
    
    def __len__(self):
        return len(self.toybox5_train)#, len(self.toybox5_test)