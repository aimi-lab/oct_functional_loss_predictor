from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from eyemod.definitions import SampleContent as sc
import eyemod.dl.data.dataset as ds
import eyemod.dl.data.transforms as tf

class BaseDataModule(LightningDataModule):

    def __init__(self, batch=16, workers=4) -> None:
        super().__init__()
        self.batch = batch
        self.workers = workers
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None

    def train_dataloader(self, shuffle=True) -> TRAIN_DATALOADERS:
        if self.train_ds is None:
            return super().train_dataloader()
        return DataLoader(self.train_ds, batch_size=self.batch, num_workers=self.workers, shuffle=shuffle)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.valid_ds is None:
            return super().val_dataloader()
        return DataLoader(self.valid_ds, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_ds is None:
            return super().test_dataloader()
        return DataLoader(self.test_ds, batch_size=self.batch, num_workers=self.workers)


class OnhDataModule(BaseDataModule):

    def __init__(self, batch = 16, workers = 4):
        super().__init__(batch, workers)

    def setup(self, stage: str):
        transform = Compose(
            [
                tf.MultiChannel(key=sc.INPUT_IMG, n_channels=3),
                tf.TransformWrapper(key=sc.INPUT_IMG, transform=Resize(size=(224, 224))),
            ]
        )

        if stage == "fit":
            self.train_ds = ds.ONHDataset(selection=list(range(200, 674)), transform=transform)
            self.valid_ds = ds.ONHDataset(selection=list(range(100, 200)), transform=transform)
        elif stage == "test":
            self.test_ds = ds.ONHDataset(selection=list(range(0, 100)), transform=transform)


if __name__ == '__main__':
    datamodule = OnhDataModule()
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()

    batch = next(iter(train_loader))
    print(len(batch))
