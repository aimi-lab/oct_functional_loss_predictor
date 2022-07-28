import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
import logging
import random
import json
import skimage.transform
from skimage import io
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class Resize:
    def __init__(self, size):
        from collections.abc import Iterable
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float64 array
        return skimage.util.img_as_ubyte(resize_image)


class OCTDataset(Dataset):
    """OCT dataset"""
    DIR_DATAFILES = Path(__file__).parent.parent.parent.joinpath("inputs", "dataset")
    DIR_ONH_IMGS = Path(__file__).parent.parent.parent.joinpath("inputs", "onh_images")
    DIR_PROJ_IMGS = Path(__file__).parent.parent.parent.joinpath("inputs", "projections")

    ID2ONH_JSON = Path(__file__).parent.parent.parent.joinpath("inputs", "dataset", "idx_to_onh.json")
    ID2PROJ_JSON = Path(__file__).parent.parent.parent.joinpath("inputs", "dataset", "idx_to_projections.json")

    def __init__(self, csv_name, target='MD', transform_image=None):

        df = pd.read_csv(OCTDataset.DIR_DATAFILES.joinpath(csv_name))
        df = df[df.slices == 61].copy()
        df["index_col"] = df["Patient ID"].astype(str) + "_" + df["Eye"] + "_" + df['vf_date'].astype(str)
        df = df.set_index("index_col")
        
        # keep_cols = [col for col in df.columns if ("Cluster" in col or col == 'MD')]
        # df = df[keep_cols]

        self.index_set = df.index.values
        self.target_set = df[target].values
        self.dataset_len = len(df)
        self.patient_set = df["Patient ID"]
        self.gs_set = df['GS']

        with open(OCTDataset.ID2ONH_JSON, "r") as infile:
            self.idx2onh_dict = json.load(infile)

        with open(OCTDataset.ID2PROJ_JSON, "r") as infile:
            self.idx2proj_dict = json.load(infile)

        self.augment_image = csv_name == 'crossval.csv'
        self.transform_image = transform_image

        # self.weights = self._get_weights_loss()
        # self.posweights = self._get_posweights()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        exam_id = self.index_set[idx]

        # FIXME: ONH images need to be mirrored?
        image_onh = io.imread(OCTDataset.DIR_ONH_IMGS.joinpath(self.idx2onh_dict[exam_id], '0.jpeg'))

        # if self.augment_image and random.random() > 0.5:
        #     image_onh = np.fliplr(image_onh)

        proj_images = []
        for img_name in [f'{no}.png' for no in list(range(7)) + [10]]:
            image_proj = io.imread(OCTDataset.DIR_PROJ_IMGS.joinpath(self.idx2proj_dict[exam_id], img_name))
            if exam_id.split('_')[1] == 'OD':
                image_proj = image_proj[:, ::-1]
            # print(image_proj.max())
            proj_images.append(image_proj[::-1, :])

        # if self.augment_image and random.random() > 0.5:
        #     proj_images = [np.fliplr(img) for img in proj_images]
        if self.augment_image:
            rot_angle = transforms.RandomRotation.get_params((-2, 2))
            proj_images = [skimage.transform.rotate(img, rot_angle) for img in proj_images]
            # scale_factor = 1 + (random.random() * 5 - 1.0) / 10.0
            # proj_images = [skimage.transform.resize(skimage.transform.rescale(img, scale_factor), img.shape) for img in proj_images]

        # proj_images = [tensor_transform(img) for img in proj_images]

        image0 = np.concatenate(proj_images[:4], axis=0) / 255
        image1 = np.concatenate(proj_images[4:], axis=0) / 255
        image2 = image_onh / 255

        image0 = skimage.transform.resize(image0, image2.shape)
        image1 = skimage.transform.resize(image1, image2.shape)

        image = np.stack([image0, image1, image2], axis=-1)

        target = self.target_set[idx].astype(np.float32)

        # FIXME: check if seed is needed in case of random augmentation
        # FIXME: check if transform has to be done before concatenating all images
        # seed = torch.randint(0, 2 ** 32, size=(1,))[0]
        if self.transform_image:
            # random.seed(seed)
            image = self.transform_image(image)

        sample = {'images': image, 'values': target, 'uuids': exam_id} #, 'center': center}
        return sample

    def get_sample(self, idx):

        sample = self[idx]

        fig, axs = plt.subplot_mosaic(
            [['im0', 'im0', 'im1', 'im1', 'im2', 'im2'],
             ['im0', 'im0', 'im1', 'im1', 'im2', 'im2'], 
             ['not', 'not', 'not', 'not', 'not', 'not']], 
             constrained_layout=True,
             figsize=(12, 4))

        # print(axs)

        # print(sample['images'][:, :, 0].shape)
        # print(sample['images'][:, :, 1].shape)
        # print(sample['images'][:, :, 2].shape)

        axs['im0'].imshow(sample['images'][0, :, :] * 0.5 + 0.5, cmap='gray')
        axs['im1'].imshow(sample['images'][1, :, :] * 0.5 + 0.5, cmap='gray')
        axs['im2'].imshow(sample['images'][2, :, :] * 0.5 + 0.5, cmap='gray')
        # print(f'{sample["uuids"]}\n{sample["values"]:.1f} dB')

        text = f'UUID = {sample["uuids"]}\nMD = {sample["values"]:.1f} dB' #+ '\n' + ''.join([f'{sample["images"][i, :, :].shape} ' for i in range(3)])
        axs['not'].text(
            0.5, 0.5, text, fontsize=30,
            horizontalalignment='center', verticalalignment='center', transform=axs['not'].transAxes
            )

        for _, ax in axs.items():
            ax.set_axis_off()

            # if label == 'not': continue
            # ax.set_title('Normal Title', fontstyle='italic')
            # ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

        fig.savefig(f'{sample["uuids"]}_sample.png')
        fig.tight_layout()
        return fig

    # def _get_weights_loss(self):
    #     labels_sum = np.sum(self.label_set, axis=0)
    #     largest_class = max(labels_sum)
    #     weights = largest_class / labels_sum
    #     weights = torch.from_numpy(weights)
    #     return weights

    # def _get_posweights(self):
    #     class_counts = np.sum(self.label_set, axis=0)
    #     pos_weights = np.ones_like(class_counts)
    #     neg_counts = [len(self.label_set) - pos_count for pos_count in class_counts]
    #     for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
    #         pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    #     return torch.tensor(pos_weights.astype(int), dtype=torch.float)


if __name__ == '__main__':

    d = OCTDataset('crossval.csv')
    # d[0]
    for ii in range(20):
        sample = d.get_sample(0)

    # d._get_posweights()
