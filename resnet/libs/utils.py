import argparse
import pathlib
import shutil
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision.transforms.functional import to_pil_image
from skimage.transform import resize
from sklearn import metrics

np.set_printoptions(precision=2)
sns.set_context('poster')

from torchcam.methods import CAM
from torchcam.utils import overlay_mask

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class RawScoresMultiOutputTarget:
    def __init__(self, out_number):
        self.out_number = out_number

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.out_number]
        return model_output[:, self.out_number]

def write_to_tb(writer, labels, scalars, iteration, phase='train'):
    for scalar, label in zip(scalars, labels):
        writer.add_scalar(f'{phase}/{label}', scalar, iteration)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source, map_location='cpu')
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)


def key_transformation(old_key):
    if old_key == 'module._fc.weight':
        return 'module._fc_new.weight'
    if old_key == 'module._fc.bias':
        return 'module._fc_new.bias'

    return old_key


def remove_module_statedict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def calculate_metrics(y_true, y_pred):

    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    mae = min(metrics.mean_absolute_error(y_true, y_pred), 5)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = max(0, metrics.r2_score(y_true, y_pred))

    dict_metrics = {'R2': r2, 'MAE': mae, 'MSE': mse}

    return dict_metrics


def run_model_on_dataset(model, data_loader, device, image_type):

    model.eval()

    preds, trues, uuid_list = [], [], []

    for data in data_loader:     

        with torch.set_grad_enabled(False):
            if image_type == 'thick':
                inputs_thick = data['images_thick'].to(device).float()
                outputs = model(inputs_thick)
            elif image_type == 'onh':
                inputs_onh = data['images_onh'].to(device).float()
                outputs = model(inputs_onh)
            else:
                assert image_type == 'combined'
                inputs_thick = data['images_thick'].to(device).float()
                inputs_onh = data['images_onh'].to(device).float()
                outputs = model(inputs_thick, inputs_onh)

        preds.append(outputs.detach().cpu().numpy())
        trues.append(data['values'])
        uuid_list.extend(data['uuids'])

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    return trues, preds, uuid_list


def eval_model(model, testset, device, save_path, image_type, dtype='test'):

    assert image_type in ['thick', 'onh', 'combined']
    
    test_trues, test_preds, test_uuids = run_model_on_dataset(model, testset, device, image_type)

    # print(test_preds.ndim)
    # print(test_preds[:, 1].ndim)
    # print(test_preds[:, 1])

    if test_trues.ndim > 1:

        test_preds = np.negative(test_preds)
        test_trues = np.negative(test_trues)
        
        df_dict = dict()
        for col in range(test_preds.shape[1]):
            df_dict[f'preds_Cluster_{(col+1):02d}'] = test_preds[:, col]
        for col in range(test_preds.shape[1]):
            df_dict[f'trues_Cluster_{(col+1):02d}'] = test_trues[:, col]
    else:

        test_preds = np.negative(test_preds.flatten())
        test_trues = np.negative(test_trues)

        df_dict = {'trues': test_trues, 'preds': test_preds}

    pd.DataFrame(df_dict, index=test_uuids).to_csv(save_path/ f'{dtype}_predictions.csv')

    if dtype != 'test':
        return

    if test_preds.ndim > 1:
        for col in range(test_preds.shape[1]):
            mae = metrics.mean_absolute_error(test_trues[:, col], test_preds[:, col])
            r2 = metrics.r2_score(test_trues[:, col], test_preds[:, col])
            text = f'MAE$_{{test}}$: {mae:.2f}\n'
            text += f'$R^2_{{test}}$: {r2:.2f}'

            fig, ax = plt.subplots(figsize=(6, 6))
            _plot_truth_pred(ax, test_trues[:, col], test_preds[:, col], text=text)
            fig.tight_layout()
            fig.savefig(save_path.joinpath(f'true_predictions_plot_only_test_Cluster_{(col+1):02d}.png'))
            fig.clf()
            plt.close()
    else:
        mae = metrics.mean_absolute_error(test_trues, test_preds)
        r2 = metrics.r2_score(test_trues, test_preds)
        text = f'MAE$_{{test}}$: {mae:.2f}\n'
        text += f'$R^2_{{test}}$: {r2:.2f}'

        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_truth_pred(ax, test_trues, test_preds, text=text)
        fig.tight_layout()
        fig.savefig(save_path.joinpath('true_predictions_plot_only_test.png'))
        fig.clf()
        plt.close()

def make_output_images(model, data_loader, device, save_path, backend = 'torchcam', **kwargs):

    if backend == 'torchcam':
        return make_output_images_torchcam(
            model=model,
            data_loader=data_loader,
            device=device,
            save_path=save_path,
            **kwargs
        )
    elif backend == 'grad-cam':
        return make_output_images_grad_cam(
            model=model,
            dataloader=data_loader,
            device=device,
            save_path=save_path,
            **kwargs
        )
    else:
        raise ValueError('Invalid make output image backend')


def make_output_images_torchcam(model, data_loader, device, save_path, image_type, n_classes):
    """
    Generation of grad cam images using the torchcam package.
    """
    
    model.eval()

    cam = CAM(model, 'module.layer4', 'module.fc_final')

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    for data in data_loader:     

        inputs = data[f'images_{image_type}'].to(device).float()
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        preds = outputs.detach().cpu().numpy()
        out_cams = [cam(class_idx=ii, normalized=False)[0].cpu().numpy() for ii in range(n_classes)]
        out_cams = np.stack(out_cams, axis=1)

        # range from -5 to 30
        scaled_cams = (out_cams + 5) / 35
        scaled_cams = np.clip(scaled_cams, 0.0, 1.0)
        print(scaled_cams.shape)

        # Iterate through batch
        for input_img, true, pred, out_cam, uuid in zip(inputs, data['values'], preds, scaled_cams, data['uuids']):
            
            norm_img = input_img.cpu().numpy()
            scaled_img = ((norm_img - norm_img.min()) / (norm_img.max() - norm_img.min()) * 255).astype(np.uint8)
            scaled_img = np.moveaxis(scaled_img, 0, -1)

            for ii in range(n_classes):

                overlay = overlay_mask(
                    to_pil_image(scaled_img), 
                    to_pil_image(resize(out_cam[ii], (8, 8), anti_aliasing=True), mode='F'),
                    colormap='rainbow_r',
                    alpha=0.5
                    )

                ax.imshow(overlay)
                ax.axis('off')
                 
                fig.tight_layout()
                fig.savefig(save_path / f'{uuid}_class{ii:02d}.png')

                ax.clear()

                plt.close(fig)

def make_output_images_grad_cam(model, dataloader, device, save_path, image_type, n_classes,):
    """
    Generation of grad cam images using the grad-cam package.
    """

    model.eval()

    target_layers = [model.module.layer4]
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    counter = 0
    for data in dataloader:     

        inputs = data[f'images_{image_type}'].to(device).float()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        preds = outputs.detach().cpu().numpy()
        outputs = outputs.sigmoid().detach().cpu().numpy()

        # Iterate through batch
        for input_img, true, pred, uuid in zip(inputs, data['values'], preds, data['uuids']):

            norm_img = input_img.cpu().numpy()[0] # first channel is grey, dataloader stacks them for resnet
            scaled_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
            bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)

            for ii in range(n_classes):

                targets = [ClassifierOutputTarget(ii)]    
                grayscale_cam = cam(input_tensor=torch.unsqueeze(input_img, 0),
                                    targets=targets, 
                                    aug_smooth=False, 
                                    eigen_smooth=False)
                
                gradcam_img = show_cam_on_image(
                    bgr_img,
                    grayscale_cam[0, :],
                    use_rgb=False,
                    colormap=cv2.COLORMAP_JET,
                    image_weight=0.8,
                )

                ax.imshow(gradcam_img)
                ax.axis('off')

                fig.tight_layout()
                fig.savefig(save_path / f'{counter}_{uuid}_class{ii:02d}.png')
                counter += 1
                ax.clear()

                plt.close(fig)


def compute_contrast(image_dir: pathlib.Path, save_dir: pathlib.Path) -> None:

    img_paths = image_dir.glob('*.png')
    contrast_list = []

    for i, img_path in enumerate(img_paths):

        # print(img_path)

        img = cv2.imread(str(img_path))

        # compute min and max of Y
        min = np.min(img)
        max = np.max(img)

        # compute contrast
        contrast = (max-min)/(max+min)
        contrast_list.append(contrast)

        if (i + 1) % 100 == 0:
            print(i + 1)

    plt.hist(contrast_list, 20)
    plt.savefig(save_dir / 'contrast.png')


def _plot_truth_pred(ax, y_true, y_pred, title=None, text=None):

    ax.scatter(y_true, y_pred, s=18, color="None", edgecolors='black', linewidths=1.2)

    ax.axline((-100, -100), slope=1., color='red', ls='--', linewidth=1.2)
    
    x = np.linspace(-100, 100, 2)
    y = np.linspace(-100, 100, 2)
    error = np.ones(2)
    error2 = np.ones(2) * 2
    plt.fill_between(x, y - error, y + error, color='red', alpha=0.15, label='$\pm 1 dB$')   
    plt.fill_between(x, y - error2, y + error2, color='red', alpha=0.15, label='$\pm 2 dB$')   
    
    ax.set_aspect('equal')

    _ = plt.ylabel("Predicted MD [dB]")
    _ = plt.xlabel("True MD [dB]")

    limone = min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1
    _ = plt.xlim(limone)
    _ = plt.ylim(limone)

    if title is not None:
        ax.set_title(title)

    if text is not None:
        plt.text(0.98, 0.02, text, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)


def plot_truth_prediction(y_true, y_pred):

    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    if y_true.ndim > 1:
        fig, axes = plt.subplots(3, 4, figsize=(6, 6))
        for ii, ax in enumerate(axes.flat[:10]):
            title = f'Cluster {ii + 1}'
            _plot_truth_pred(ax, y_true[:, ii], y_pred[:, ii], title)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_truth_pred(ax, y_true, y_pred)

    return fig
