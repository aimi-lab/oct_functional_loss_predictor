import torch
import numpy as np
import pandas as pd
import math
from sklearn import metrics
# from scipy.interpolate import interp1d
import argparse
import shutil
import json
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import pathlib
import seaborn as sns
from skimage.transform import resize

np.set_printoptions(precision=2)
sns.set_context('poster')

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
# from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
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


# def create_output_images(model, data_loader, device):
    
#     if isinstance(data_loader.dataset, torch.utils.data.Subset):
#         dataset = data_loader.dataset.dataset

# [ClassifierOutputTarget(category) for category in target_categories]


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


def make_output_images(model, data_loader, device, save_path, image_type, n_classes):
    
    model.eval()

    target_layers = [model.module.layer4[-1]] # changed from layer 4
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    for data in data_loader:     

        # preds, trues, uuid_list = [], [], []
        inputs = data[f'images_{image_type}'].to(device).float()
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        preds = outputs.detach().cpu().numpy()
        
        # Iterate through batch
        for input_img, true, pred, uuid in zip(inputs, data['values'], preds, data['uuids']):
            
            if image_type == 'onh':
                norm_img = input_img.cpu().numpy()[0] # first channel is grey, dataloader stacks them for resnet
                scaled_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
                bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)
            else:
                norm_img = input_img.cpu().numpy()[0] # added [0]
                scaled_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
                # scaled_img = np.moveaxis(scaled_img, 0, -1) # removed
                bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB) #cv2.COLOR_RGB2BGR

            for ii in range(n_classes):

                target = [RawScoresMultiOutputTarget(ii)]    
                grayscale_cam = cam(input_tensor=torch.unsqueeze(input_img, 0),
                    targets=target, 
                    aug_smooth=False, # if image_type == 'onh' else False, 
                    eigen_smooth=True) # if image_type == 'onh' else False)
                    
                gradcam_img = show_cam_on_image(bgr_img, grayscale_cam[0, :], use_rgb=True, colormap=cv2.COLORMAP_CIVIDIS, image_weight=0.7)

                # if image_type == 'onh':
                # gradcam_imgs = [gradcam_img]
                # labels = ['']
                # axess = [axes]
                # else:
                #     # print(gradcam_img.shape)
                #     gradcam_img = resize(gradcam_img, (512*7, 512), anti_aliasing=True)
                #     # print(gradcam_img.shape)
                #     gradcam_imgs = np.split(gradcam_img, 7, axis=0)
                #     labels = ['1', 'RNFL', 'GCL', 'HIHI', '5', '6', '7', '8']
                #     axess = axes.flat

                # for gradcam_layer, axx, label in zip(gradcam_imgs, axess, labels):
                ax.imshow(gradcam_img)
                ax.axis('off')
                # ax.text(0.02, 0.02, '',
                #         color='white', # fontsize='x-small',
                #         horizontalalignment='left', 
                #         verticalalignment='bottom', 
                #         transform=ax.transAxes)

                # axess[-1].text(0.5, -0.1, f'UUID:{uuid}\nTrue MD: {float(-1*true):.2f} dB\nPred MD: {float(-1*pred):.2f} dB',
                #             horizontalalignment='center', 
                #             verticalalignment='top', 
                #             transform=axess[-1].transAxes)
                ax.axis('off')

                # if image_type == 'thick':
                #     fig.suptitle(f'UUID: {uuid}\nTrue MD: {float(-1*true):.2f} dB\nPred MD: {float(-1*pred):.2f} dB', fontsize=16)
                
                fig.tight_layout()
                fig.savefig(save_path / f'{uuid}_class{ii:02d}.png')

                # for axx in axess:
                ax.clear()

                plt.close(fig)
        
        # break


def compute_contrast(image_dir: pathlib.Path) -> None:

    img_paths = image_dir.glob('*.png')
    contrast_list = []

    for i, img_path in enumerate(img_paths):

        # print(img_path)

        img = cv2.imread(str(img_path))
        # Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

        # compute min and max of Y
        min = np.min(img)
        max = np.max(img)

        # compute contrast
        contrast = (max-min)/(max+min)
        contrast_list.append(contrast)

        if (i + 1) % 100 == 0:
            print(i + 1)

    plt.hist(contrast_list, 20)
    plt.savefig('hihi.png')


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

    # fig.savefig(os.path.join(save_dir, 'true_predictions_plot_only_test.png'))
    # fig.clf()
    # plt.close()
    return fig


if __name__ == '__main__':

    img_dir = pathlib.Path(__file__).parent.parent.absolute() / 'inputs' / 'slices'
    compute_contrast(img_dir)

