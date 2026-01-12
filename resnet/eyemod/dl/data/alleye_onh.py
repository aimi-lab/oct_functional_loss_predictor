from pathlib import Path
from collections.abc import Iterable
import tarfile

import glob

import torch
import webdataset as wds
import torchvision.transforms.v2 as T


class AlleyeOnh(wds.DataPipeline):
    SPLIT_PREFIX = 'split-'
    def __init__(self, root, split, transform=None, target_transform=None, shuffle: bool = False):
        super().__init__()

        self.root = Path(root)
        split_dirs = self._get_split_dirs(split)
        
        shards = self._get_shards(split_dirs)

        self.len = self.count_npy_files(shards)

        self.append(wds.SimpleShardList(shards))

        if shuffle:
            self.append(wds.shuffle(1000))

        # at this point, we have an iterator over the shards assigned to each worker
        self.append(wds.tarfile_to_samples())

        self.append(wds.decode())
        self.append(wds.to_tuple('npy', 'json'))
        self.append(wds.map_tuple(self._preprocess_img, self._extract_meta))
        
        if transform:
             self.append(wds.map_tuple(transform, self._identity))
        
        if target_transform:
            self.append(wds.map_tuple(self._identity, target_transform))

    def __len__(self):
        return self.len
    
    @staticmethod
    def _identity(x):
        return x

    def _get_split_dirs(self, split: list | int):
        
        if not isinstance(split, Iterable):
            split = [split]

        split_dirs = list(self.root.glob(self.SPLIT_PREFIX + '*'))
        out = []
        for dir in split_dirs:
            s = int(dir.name.removeprefix(self.SPLIT_PREFIX))
            if s in split:
                out.append(dir)
        return out

    def _get_shards(self, dirs: list[Path]) -> list[str]:
        shards = []
        for d in dirs:
            shards.extend(d.glob('*.tar'))
        # convert to string for later use with SimpleShardList 
        shards = [str(s) for s in shards]
        return shards

    def count_npy_files(self, shard_paths: list[str]) -> int:
        total_count = 0
        for shard_path in shard_paths:
            with tarfile.open(shard_path, "r") as tar:
                npy_count = sum(
                    1 for member in tar.getmembers()
                    if member.isfile() and member.name.endswith(".npy")
                )
                total_count += npy_count
        return total_count


    @staticmethod
    def _preprocess_img(img):
        img = torch.from_numpy(img)
        img = T.ToDtype(torch.float32, scale=True)(img)
        # add channel dimension
        img = img.unsqueeze(0)
        return img
    
    @staticmethod
    def _extract_meta(json: dict):
        meta = {key: json.get(key) for key in ['heyex_id_anon', 'laterality', 'acquisition_date']}
        return meta
         
        
def main():
    data_paths = []
    data_paths.extend(glob.glob("data/processed/alleye-onh/split-000/*.tar"))
    data_paths.extend(glob.glob("data/processed/alleye-onh/split-001/*.tar"))

    transform = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.5], std=[0.5])  # adjust for grayscale or RGB
    ])

    data_paths = 'data/processed/alleye-onh/split-{000..004}/split-{000..004}-00000.tar'

    ds = AlleyeOnh(root = 'in/data/alleye-onh', split=[0, 1, 2, 3], transform=transform, target_transform=None)



    # loader = DataLoader(ds, batch_size=4, num_workers=2)

    for i, (image, meta) in enumerate(ds):
        print(f"\Sample {i + 1}")
        print(f"Image type: {type(image)}, shape: {image.size if isinstance(image, list) else image.shape}")
        print(f"Label type: {type(meta)}, sample: {meta}")


if __name__ == '__main__':
    main()
