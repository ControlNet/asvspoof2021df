import os
from typing import Optional, Union, List, Dict

import numpy as np
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from data_utils import genSpoof_list


class ASVspoof2019Trill(Dataset):

    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X = np.load(self.base_dir + "trill_emb_layer19/" + key + ".npy").astype("float32").mean(axis=0)
        x_inp = Tensor(X)
        if self.labels is not None:
            y = self.labels[key]
        else:
            y = key
        return x_inp, y


class ASVspoof2019TrillMean(ASVspoof2019Trill):

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X = np.load(self.base_dir + "trill_emb_layer19_mean/" + key + ".npy").astype("float32")
        x_inp = Tensor(X)
        if self.labels is not None:
            y = self.labels[key]
        else:
            y = key
        return x_inp, y


class ASVspoof2019TrillDataModule(LightningDataModule):

    def __init__(self, batch_size=128, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.d_label_trn, self.file_train = genSpoof_list(
            dir_meta=os.path.join(
                "dataset/ASVspoof_DF_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
            is_train=True, is_eval=False)
        print('no. of training trials', len(self.file_train))

        self.d_label_dev, self.file_dev = genSpoof_list(
            dir_meta=os.path.join(
                'dataset/ASVspoof_DF_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
            is_train=False, is_eval=False)
        print('no. of validation trials', len(self.file_dev))

        self.file_eval = genSpoof_list(
            dir_meta='dataset/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt', is_train=False,
            is_eval=True)
        print('no. of eval trials', len(self.file_eval))

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ASVspoof2019Trill(list_IDs=self.file_train, labels=self.d_label_trn,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_train/'))
        self.val_dataset = ASVspoof2019Trill(list_IDs=self.file_dev,
            labels=self.d_label_dev,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_dev/'))
        self.test_dataset = ASVspoof2019Trill(list_IDs=self.file_eval, labels=None,
            base_dir='dataset/ASVspoof2021_DF_eval/')

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
            num_workers=self.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class ASVspoof2019TrillMeanDataModule(ASVspoof2019TrillDataModule):

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ASVspoof2019TrillMean(list_IDs=self.file_train, labels=self.d_label_trn,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_train/'))
        self.val_dataset = ASVspoof2019TrillMean(list_IDs=self.file_dev,
            labels=self.d_label_dev,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_dev/'))
        self.test_dataset = ASVspoof2019TrillMean(list_IDs=self.file_eval, labels=None,
            base_dir='dataset/ASVspoof2021_DF_eval/')


class ASVspoof2019TrillMeanDataModule0(ASVspoof2019TrillDataModule):

    def __init__(self, batch_size=128, num_workers=1):
        super().__init__(batch_size, num_workers)
        self.train_dataset_idx = range(2580, 25380)
        self.val_dataset_ids = range(2548, 24844)

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = Subset(ASVspoof2019TrillMean(list_IDs=self.file_train, labels=self.d_label_trn,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_train/')), self.train_dataset_idx)
        self.val_dataset = Subset(ASVspoof2019TrillMean(list_IDs=self.file_dev,
            labels=self.d_label_dev,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_dev/')), self.val_dataset_ids)
        self.test_dataset = ASVspoof2019TrillMean(list_IDs=self.file_eval, labels=None,
            base_dir='dataset/ASVspoof2021_DF_eval/')
    
class ASVspoof2019TrillMeanDataModule1(ASVspoof2019TrillDataModule):

    def __init__(self, batch_size=128, num_workers=1):
        super().__init__(batch_size, num_workers)
        self.train_dataset_idx = range(2580)
        self.val_dataset_ids = range(2548)

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = Subset(ASVspoof2019TrillMean(list_IDs=self.file_train, labels=self.d_label_trn,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_train/')), self.train_dataset_idx)
        self.val_dataset = Subset(ASVspoof2019TrillMean(list_IDs=self.file_dev,
            labels=self.d_label_dev,
            base_dir=os.path.join('dataset/ASVspoof2019_LA_dev/')), self.val_dataset_ids)
        self.test_dataset = ASVspoof2019TrillMean(list_IDs=self.file_eval, labels=None,
            base_dir='dataset/ASVspoof2021_DF_eval/')
