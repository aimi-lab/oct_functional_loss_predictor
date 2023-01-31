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
import imgaug.augmenters as iaa
import cv2

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
    DIR_THICK_IMGS = Path(__file__).parent.parent.parent.joinpath("inputs", "thicknesses")

    ID2ONH_JSON = Path(__file__).parent.parent.parent.joinpath("inputs", "dataset", "idx_to_onh.json")
    ID2THICK_JSON = Path(__file__).parent.parent.parent.joinpath("inputs", "dataset", "idx_to_thicknesses.json")
    augment_image = False

    def __init__(self, csv_name, target, transform_image, thick_or_onh='thick'):

        assert thick_or_onh in ['thick', 'onh', 'combined'], 'thick_or_onh: choice not valid'
        
        df = pd.read_csv(OCTDataset.DIR_DATAFILES.joinpath(csv_name))
        # df = df[df.slices == 61].copy()
        df["index_col"] = df["Patient ID"].astype(str) + "_" + df["Eye"] + "_" + df['vf_date'].astype(str)
        df = df.set_index("index_col")

        self.set_target(target)
        
        # keep_cols = [col for col in df.columns if ("Cluster" in col or col == 'MD')]
        # df = df[keep_cols]

        self.index_set = df.index.values
        self.target_set = df[self._target].values
        self.dataset_len = len(df)
        self.patient_set = df["Patient ID"]
        self.gs_set = df['GS']
        self.weight_set = self._get_weights_loss()

        with open(OCTDataset.ID2ONH_JSON, "r") as infile:
            self.idx2onh_dict = json.load(infile)

        with open(OCTDataset.ID2THICK_JSON, "r") as infile:
            self.idx2thick_dict = json.load(infile)

        self.transform_image = transform_image
        self.return_imgs = thick_or_onh

        # self.posweights = self._get_posweights()

    @staticmethod
    def rgba2rgb(rgba, background=(0,0,0)):
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros( (row, col, 3), dtype='float32' )
        r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

        a = np.asarray( a, dtype='float32' ) / 255.0

        R, G, B = background

        rgb[:,:,0] = r * a + (1.0 - a) * R
        rgb[:,:,1] = g * a + (1.0 - a) * G
        rgb[:,:,2] = b * a + (1.0 - a) * B

        return np.asarray( rgb, dtype='uint8' )

    def set_target(self, target):
        if target == 'MD':
            self._target = 'MD'
        elif target == 'clusters':
            self._target = [f'Cluster {ii}' for ii in range(1, 11)]

    def __len__(self):
        return self.dataset_len

    def get_thickmap_images(self, exam_id):

        proj_images = []
        # for img_name in [f'{no}.png' for no in list(range(1, 7)) + [10]]:
        for img_name in [f'{no}.png' for no in [10]]:
            image_thick = io.imread(OCTDataset.DIR_THICK_IMGS.joinpath(self.idx2thick_dict[exam_id], img_name))

            # image_thick = OCTDataset.rgba2rgb(image_thick)
            image_thick = cv2.cvtColor(image_thick, cv2.COLOR_RGBA2GRAY)
            
            if exam_id.split('_')[1] == 'OD':
                image_thick = image_thick[:, ::-1]
            proj_images.append(image_thick[::-1, :])

        if OCTDataset.augment_image:
            aug = iaa.Sequential([  
                    # iaa.Fliplr(0.5),
                    iaa.Affine(
                        scale={"x": (0.99, 1.01), "y": (0.99, 1.01)},
                        translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                        rotate=(-0.5, 0.5),
                    ),
                    # iaa.LinearContrast((0.95, 1.05))
                ])
            aug_det = aug.to_deterministic()
            proj_images = [aug_det.augment_image(img) for img in proj_images]

        image0 = np.concatenate(proj_images, axis=0) / 256
        # image0 = np.concatenate([img[:, :, 0] for img in proj_images], axis=0) / 256
        # image1 = np.concatenate([img[:, :, 1] for img in proj_images], axis=0) / 256
        # image2 = np.concatenate([img[:, :, 2] for img in proj_images], axis=0) / 256

        image = np.stack([image0, image0, image0], axis=-1)
        # image = np.stack([image0, image1, image2], axis=-1)
        return image

    def get_onh_image(self, exam_id):

        image_onh = io.imread(OCTDataset.DIR_ONH_IMGS.joinpath(self.idx2onh_dict[exam_id], '0.jpeg'))
        image_onh = image_onh / 256

        # FIXME: do not mirror when clusters (?). Flip only if Left/Right eye
        # it seems that not flipping ONH images performs better
        # if exam_id.split('_')[1] == 'OD':
        #     image_onh = image_onh[:, ::-1]

        if OCTDataset.augment_image:
            # FIXME: consider reduction of augmentation parameters
            aug = iaa.Sequential([  
                                # iaa.Fliplr(0.5),
                                iaa.Affine(
                                    scale={"x": (0.99, 1.09), "y": (0.99, 1.01)},
                                    translate_percent={"x": (-0.02, 0.02), "y": (-0.05, 0.02)},
                                    rotate=(-0.8, 0.8),
                                ),
                                # iaa.LinearContrast((0.95, 1.05))
                            ])
            image_onh = aug.augment_image(image_onh)
            # image_onh = np.fliplr(image_onh)

        image = np.stack([image_onh, image_onh, image_onh], axis=-1)
        return image

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        exam_id = self.index_set[idx]
        target = self.target_set[idx].astype(np.float32)
        weight = self.weight_set[idx]

        if self.return_imgs == 'thick':
            image_thick = self.get_thickmap_images(exam_id)
            image_thick = self.transform_image(image_thick)
            image_onh = 'dummy'
        elif self.return_imgs == 'onh':
            image_onh = self.get_onh_image(exam_id)
            image_onh = self.transform_image(image_onh)
            image_thick = 'dummy'
        else:
            image_thick = self.get_thickmap_images(exam_id)
            image_onh = self.get_onh_image(exam_id)
            image_onh = self.transform_image(image_onh)
            image_thick = self.transform_image(image_thick)

        sample = {'images_thick': image_thick, 'images_onh': image_onh, 'values': target, 'uuids': exam_id, 'weights': weight} #, 'center': center}
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

        axs['im0'].imshow(sample[f'images_{self.return_imgs}'][0, :, :] * 0.5 + 0.5, cmap='gray')
        axs['im1'].imshow(sample[f'images_{self.return_imgs}'][1, :, :] * 0.5 + 0.5, cmap='gray')
        axs['im2'].imshow(sample[f'images_{self.return_imgs}'][2, :, :] * 0.5 + 0.5, cmap='gray')
        # print(f'{sample["uuids"]}\n{sample["values"]:.1f} dB')

        # text = f'UUID = {sample["uuids"]}\nMD = {sample["values"]:.1f} dB' #+ '\n' + ''.join([f'{sample["images"][i, :, :].shape} ' for i in range(3)])
        text = f'UUID = {sample["uuids"]}' #+ '\n' + ''.join([f'{sample["images"][i, :, :].shape} ' for i in range(3)])
        axs['not'].text(
            0.5, 0.5, text, fontsize=30,
            horizontalalignment='center', verticalalignment='center', transform=axs['not'].transAxes
            )

        for _, ax in axs.items():
            ax.set_axis_off()

            # if label == 'not': continue
            # ax.set_title('Normal Title', fontstyle='italic')
            # ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

        # fig.savefig(f'{sample["uuids"]}_sample.png')
        fig.tight_layout()
        return fig

    def _get_weights_loss(self):
        value_counts = self.gs_set.value_counts()
        weights_dict = (value_counts.max() / value_counts).to_dict()
        weights = self.gs_set.map(weights_dict)
        # weights = torch.from_numpy(weights.values)
        return weights

    # def _get_posweights(self):
    #     class_counts = np.sum(self.label_set, axis=0)
    #     pos_weights = np.ones_like(class_counts)
    #     neg_counts = [len(self.label_set) - pos_count for pos_count in class_counts]
    #     for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
    #         pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    #     return torch.tensor(pos_weights.astype(int), dtype=torch.float)


if __name__ == '__main__':

    d = OCTDataset('crossval.csv', target='MD', transform_image=None)
    # d[0]
    for ii in range(20):
        sample = d.get_sample(0)

    # d._get_posweights()
