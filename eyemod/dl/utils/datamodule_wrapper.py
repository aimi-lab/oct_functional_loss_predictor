from torch.utils.data import DataLoader
from lightning import LightningDataModule


class DataModuleWrapper(LightningDataModule):
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
    ):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    @classmethod
    def fromdict(cls, dataloaders: dict):
        return cls(
            train_loader=dataloaders.get("train", None),
            val_loader=dataloaders.get("val", None),
            test_loader=dataloaders.get("test",None),
        )

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
