from pathlib import Path
from collections.abc import Mapping, Sequence
import logging
import io
import time
import shutil

import wandb
import torch
import numpy as np 
import pandas as pd
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from lightning import Callback
from lightning.pytorch.loggers import WandbLogger

from torchmetrics import MeanAbsoluteError, R2Score

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from aimitools.visual_field import VisualField

from eyemod.data.glaucoma_stage import GlaucomaStage

LOGGER = logging.getLogger(__name__)

class Results():
    """
    Utility class to accumulate results.
    """
    def __init__(self):
        self.data: list[dict] = []
    
    def __len__(self):
        return len(self.data)
        
    def append(self, data: dict):
        data = self._flatten_dict(data)
        data = self._detach_dict(data)

        self.data.append(data)
    
    def clear(self):
        self.data.clear()

    def collate_results(self) -> dict[str, np.ndarray]:
        results = self._to_dict_of_list(self.data)

        for k in results.keys():
            results[k] = np.concatenate(results[k], axis=0) if len(results[k]) > 0 else None

        return results

    def _detach_dict(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Detach all tensors stored in the dictionary
        """
        for k in x.keys():
            if isinstance(x[k], torch.Tensor):
                x[k] = x[k].detach().cpu()
        return x
        

    def _flatten_dict(self, input: dict, parent_key='', sep='.'):
        """
        Flatten a nested dictionary.
        
        Args:
            input: Dictionary to flatten
            parent_key: Key prefix for nested keys
            sep: Separator between parent and child keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in input.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _to_dict_of_list(self, _dicts: list[dict]) -> dict[list]:
        """
        Convert list of dictionaries to dictionary of lists
        """
        if len(_dicts) == 0:
            return {}
        keys = _dicts[0].keys()
        out = {k:[] for k in keys}
        for entry in _dicts:
            for k in keys:
                out[k].append(entry[k])
        return out


class GatherCallback(Callback):
    """
    Base callback class which gathers the output data during the specified stages.
    The gathered data can be used at the end of the epoch.

    store interval of -1 indicates storing last only
    """
    VALID_STAGES = ['train', 'val', 'test']
    def __init__(self, stages: str | list[str], store_interval: int = 10, keys_to_store: list[str] = None):
        super().__init__()

        if isinstance(stages, str):
            stages = [stages]
        self.stages = stages

        self.is_storing_on = True
        self.store_interval = store_interval
        self.keys_to_store = set(keys_to_store) if isinstance(keys_to_store, Sequence) else None

        self.data: dict[str, Results] = None

    def setup(self, trainer, pl_module, stage):
        self._init_data()

    def _init_data(self):
        self.data = {}
        for s in self.stages:
            self.data[s] = Results()

    def _toggle_storing(self, pl_module):
        current_epoch = pl_module.current_epoch
        last_epoch = pl_module.trainer.max_epochs - 1

        # always store the last epoch
        if current_epoch == last_epoch:
            self.is_storing_on = True
        elif self.store_interval > 0 and current_epoch % self.store_interval == 0:
            self.is_storing_on = True
        else:
            self.is_storing_on = False

    def _store_output(self, stage: str, outputs):
        if stage in self.stages and self.is_storing_on:
            outputs = self._filter_output(outputs)
            self.data[stage].append(outputs)

    def _filter_output(self, outputs):
        if self.keys_to_store is not None:
            return {k: outputs[k] for k in outputs.keys() & self.keys_to_store}
        else:
            return outputs

    def _clear_output(self, stage: str):
        if stage in self.stages and self.is_storing_on:
            self.data[stage].clear()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._toggle_storing(pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx = 0):
        self._toggle_storing(pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx = 0):
        self._toggle_storing(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._store_output('train', outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._store_output('val', outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._store_output('test', outputs)


class ScatterPlotCallback(GatherCallback):
    """
    Callback to generate scatter plots of regression tasks.
    Plots prediction versus target.
    """
    def __init__(
        self,
        stages: list[str] | str,
        x_data: str = 'target',
        y_data: str = 'output',
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        plot_interval: int = 10,
        output_names: list[str] = None,
        plot_mode: str = 'combined',
        reduction: str = None
    ):
        super().__init__(stages=stages, store_interval=plot_interval, keys_to_store=['output', 'target', 'meta'])

        self.x_data = x_data
        self.y_data = y_data

        self.title = title if title else ''
        self.x_label = x_label if x_label else 'expected'
        self.y_label = y_label if y_label else 'predicted'

        self.output_names = output_names

        self.plot_mode = plot_mode
        if self.plot_mode not in ['combined', 'separate']:
            raise ValueError("plot_mode must be 'combined', 'separate'")
        
        self.reduction = reduction
        if self.reduction is not None and self.reduction not in ['mean', 'none']:
            raise ValueError("reduction must be 'mean', 'none'")

    @staticmethod
    def _truncate_strings(df: pd.DataFrame, max_len: int = 12) -> pd.DataFrame:
        def trunc_fn(x):
            if isinstance(x, str):
                return x[:max_len] + '...' if len(x) > max_len else x
            else:
                return x
        return df.map(trunc_fn)

    def _log_scatter_plot(self, stage: str, trainer):

        if stage in self.stages and self.is_storing_on:

            if len(self.data[stage]) == 0:
                LOGGER.warning('No data to plot')
                return

            data = self._prepare_data(self.data[stage])

            if data[self.x_data].ndim == 1:
                self._log_single_output(data, stage=stage, trainer=trainer)
            elif data[self.x_data].ndim == 2:
                self._log_multi_output(data, stage=stage, trainer=trainer)
            else:
                raise ValueError('Cannot scatter plot multi dimensional output data')

    def _prepare_data(self, res: Results) -> dict:

        data = res.collate_results()
        for k in data.keys():
            data[k] = np.array(data[k]).squeeze()

        if self.reduction == 'mean':
            for k in [self.x_data, self.y_data]:
                data[k] = np.mean(data[k], axis=1)

        # remove data that has more than two dimensions
        to_remove = [k for k, v in data.items() if v.ndim > 2]
        [data.pop(k) for k in to_remove]

        return data

    def _log_single_output(self, data: dict, stage: str, trainer):

        assert len(data[self.y_data].shape) == 1, "Output must be one dimensional for scatter plotting"
        assert len(data[self.x_data].shape) == 1, "Target must be one dimensional for scatter plotting"

        df = pd.DataFrame.from_dict(data)

        fig = self._create_plot(df, self.x_label, self.y_label)

        plot_name = f"{stage}_scatter_{self.title}" if self.title else f"{stage}_scatter"
        trainer.logger.experiment.log({f"callback/{plot_name}": fig})
    

    def _log_multi_output(self, data: dict, stage: str, trainer):
        
        pred_data = data.pop(self.y_data)
        target_data = data.pop(self.x_data)

        df_meta = pd.DataFrame.from_dict(data)

        n_samples, n_outputs = pred_data.shape

        if self.output_names is None:
            self.output_names = [f'output_{i}' for i in range(n_outputs)]

        if self.plot_mode == 'combined':
            # unpack data to long format and add id variable indicating the output dimension
            # use id variable to color the dots depending on the output dimension
            id_col = 'output_id'

            df_pred = pd.DataFrame(pred_data, columns=self.output_names)
            df_pred = df_pred.melt(value_name=self.y_data, var_name=id_col, ignore_index=False)

            df_target = pd.DataFrame(target_data, columns=self.output_names)
            df_target = df_target.melt(value_name=self.x_data, var_name=id_col, ignore_index=False)
            df_target = df_target.drop(columns=id_col)

            df_data = pd.concat([df_pred, df_target], axis = 1)
            df_data = df_meta.join(df_data, how='inner')

            fig = self._create_plot(df_data, self.x_label, self.y_label, color_col=id_col)

            plot_name = f"{stage}_scatter_{self.title}" if self.title else f"{stage}_scatter"
            trainer.logger.experiment.log({f"callback/{plot_name}": fig})
            return

        elif self.plot_mode == 'separate':
            for i, out_name in enumerate(self.output_names):
                pred = pred_data[:, i]
                target = target_data[:, i]
                data = np.vstack([pred, target]).T

                df_data = pd.DataFrame(data=data, columns=[self.y_data, self.x_data])
                df_data = df_meta.join(df_data)

                fig = self._create_plot(df_data, self.x_label, self.y_label)

                plot_name = f"{stage}_scatter_{self.title}_{out_name}" if self.title else f"{stage}_scatter_{out_name}"
                trainer.logger.experiment.log({f"callback/{plot_name}": fig})
            return
        
        else:
            raise ValueError(f'Unknown plot mode: {self.plot_mode}')       
        
    def _create_plot(self, df: pd.DataFrame, x_label: str, y_label: str, color_col: str = None) -> go.Figure:
        df = self._truncate_strings(df, max_len=12)

        if color_col:
            fig = px.scatter(
                df,
                x=self.x_data,
                y=self.y_data,
                color=color_col,
                hover_data=[col for col in df.columns if col != color_col],
            )
        else:
            fig = px.scatter(df, x=self.x_data, y=self.y_data, hover_data=df.columns)

        fig.add_trace(self._create_diagonal_line(df[self.x_data]))

        fig.update_layout(
            title=self.title if self.title else '',
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=False,
            hoverlabel={'font_size': 10}
        )
        return fig

    def _create_diagonal_line(self, x_data: np.ndarray) -> go.Scatter:
        """Create diagonal reference line"""
        return go.Scatter(
            x=[min(x_data), max(x_data)],
            y=[min(x_data), max(x_data)],
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='y=x',
            showlegend=False,
            hoverinfo='skip'
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_scatter_plot('train', trainer)
        self._clear_output('train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_scatter_plot('val', trainer)
        self._clear_output('val')

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_scatter_plot('test', trainer)
        self._clear_output('test')


class OutputWriter(GatherCallback):
    def __init__(self, stages: str | list[str], run_dir: str, write_interval: int = 10, dir_name: str = 'output', out_type: str = 'parquet', reduction: str = None):
        super().__init__(stages=stages, store_interval=write_interval, keys_to_store=['output', 'target', 'meta'])

        self.output_dir = Path(run_dir) / dir_name

        self.out_type = out_type
        if self.out_type is not None and self.out_type not in ['csv', 'parquet']:
            raise ValueError("out_type must be 'csv' or 'parquet'")

        self.reduction = reduction
        if self.reduction is not None and self.reduction not in ['mean', 'none']:
            raise ValueError("reduction must be 'mean', 'none'")

    def setup(self, trainer, pl_module, stage):
        self.output_dir.mkdir(parents=True)
        self._init_data()
    
    
    def _write_output(self, stage: str, pl_module):
        if stage in self.stages and self.is_storing_on:
            global_step = pl_module.global_step
            epoch = pl_module.current_epoch

            filename = f'output_{stage}_ep{epoch}_gbl-step{global_step}'
            out_path = self.output_dir / filename

            data = self.data[stage].collate_results()
            data = self._apply_reduction(data)

            if self.out_type == 'csv':
                out_path = out_path.with_suffix('.csv')
                data = self._prepare_for_csv(data)
                data = pd.DataFrame(data)
                data.to_csv(out_path)

            elif self.out_type == 'parquet':
                out_path = out_path.with_suffix('.parquet')
                data = self._prepare_for_parquet(data)
                data = pd.DataFrame(data)
                data.to_parquet(out_path)

    def _apply_reduction(self, data: dict) -> dict:

        if self.reduction == 'mean':
            for k in ['output', 'target']:
                data[k] = np.mean(data[k], axis=1)
        return data


    def _prepare_for_csv(self, data: dict) -> dict:
        """
        Convert multidimensional data to 1D format suitable for storing in CSV format.
        """
        processed = {}

        for key, value in data.items():
            if value is None:
                continue

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    # Scalar
                    processed[key] = [value.item()]
                elif value.ndim == 1:
                    # 1D array
                    processed[key] = value
                elif value.ndim == 2:
                    # 2D array - create separate columns for each feature
                    for i in range(value.shape[1]):
                        processed[f'{key}_{i}'] = value[:, i]
                else:
                    try:
                        processed[key] = str(value.tolist())
                    except:
                        raise NotImplementedError('Saving of higher dim data to CSV is not implemented')
            else:
                processed[key] = value
        return processed

    def _prepare_for_parquet(self, data: dict) -> dict:
        processed = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                processed[key] = value.tolist()
            else:
                processed[key] = value
        return processed

    
    def on_train_epoch_end(self, trainer, pl_module):
        self._write_output('train', pl_module)
        self._clear_output('train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self._write_output('val', pl_module)
        self._clear_output('val')
    
    def on_test_epoch_end(self, trainer, pl_module):
        self._write_output('test', pl_module)
        self._clear_output('val')

class DatasetSummary(Callback):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self._summarize(trainer)

    def _summarize(self, trainer):
        print('\nDataset Summary')
        splits = ('train', 'val', 'test')
        for s in splits:
            loader = getattr(trainer.datamodule, f'{s}_dataloader')()
            if loader is not None:
                dataset = loader.dataset
                name = dataset.__class__.__name__
                print(f'{s.capitalize():<5} Dataset: {name} Size: {len(dataset)}')


class FreezingSummary(Callback):
    """
    Callback to print the status of requires_grad of all used modules.
    """
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self._summarize(trainer)
    
    def _summarize(self, trainer):
        # gather information
        data = {
            'Name': [],
            'requires_grad': []
        }

        for name, param in trainer.model.named_parameters():
            data['Name'].append(name)
            data['requires_grad'].append(param.requires_grad)
    
        col_widths = []
        for key, values in data.items():
            max_width = max(len(key),max([len(str(x)) for x in values]))
            col_widths.append(max_width)

        
        header = [f'{' ':4}']
        header.extend([f'{key:<{width}}' for key, width in zip(data.keys(), col_widths)])
        header = ' | '.join(header)

        # print table
        print('\nFreezing Summary')
        print('-' * len(header))
        print(header)
        print('-' * len(header))

        for i in range(len(data['Name'])):
            line = [f'{i:<4}']
            line.extend([f'{str(x[i]):<{width}}' for x, width in zip(data.values(), col_widths)])
            line = ' | '.join(line)
            print(line)
        print('-' * len(header))


class EpochTimer(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.start_time
        trainer.logger.log_metrics({"epoch_duration":epoch_duration})


class ImageViewer(GatherCallback):
    def __init__(self, stages, store_interval = 10, num_images: int = 5, class_labels: dict[int, str] = None):
        super().__init__(stages, store_interval)

        self.num_images = num_images
        self.image_indices = {}
        self.batch_offsets = {}

        if isinstance(class_labels, Mapping):
            self.class_labels = dict(class_labels)

    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)

        rng = np.random.default_rng(seed=42)
        for stage in self.stages:
            dataset_size = self._get_dataset_size(trainer=trainer, stage=stage)
            indices = np.arange(dataset_size)
            rng.shuffle(indices)
            selected_indices = indices[:self.num_images] if self.num_images < dataset_size else indices[:dataset_size]
            self.image_indices[stage] = selected_indices

            self.batch_offsets[stage] = 0

    def _get_dataset_size(self, trainer, stage: str) -> int:
        """Get the number of samples in the dataset for a given stage."""
        if stage == 'train':
            return len(trainer.datamodule.train_dataloader().dataset)
        elif stage == 'val':
            return len(trainer.datamodule.val_dataloader().dataset)
        elif stage == 'test':
            return len(trainer.datamodule.test_dataloader().dataset)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def on_sanity_check_start(self, trainer, pl_module):
        """
        Define separate set of indices for sanity checking.
        Only subset of dataset is used, thus selecting indices based on the whole dataset would not work.
        """

        self.image_indices['val_temp'] = self.image_indices['val']

        num_sanity_batches = trainer.num_sanity_val_steps
        val_dataloader = trainer.datamodule.val_dataloader()
        batch_size = val_dataloader.batch_size

        dataset_size = batch_size * num_sanity_batches
        indices = np.arange(dataset_size)

        self.image_indices['val'] = indices[:self.num_images] if self.num_images < dataset_size else indices[:dataset_size]

    def on_sanity_check_end(self, trainer, pl_module):
        self.image_indices['val'] = self.image_indices.pop('val_temp')

    def _store_output(self, stage: str, outputs):
        """
        Overwrite store function to only store the selected images.
        Otherwise out of memory error can arise.
        """
        if stage in self.stages and self.is_storing_on:

            pred = outputs['output']
            target = outputs['target']
            mask = outputs.get('mask', None)
            meta = outputs.get('meta', None)

            batch_size = len(outputs['output'])

            # Get batch indices for this stage
            batch_start = self.batch_offsets[stage]
            batch_end = batch_start + batch_size
            batch_indices = np.arange(batch_start, batch_end)

            # Check which indices from this batch should be stored
            target_indices = self.image_indices.get(stage, [])
            store_mask = np.isin(batch_indices, target_indices)

            if not store_mask.any():
                self.batch_offsets[stage] = batch_end
                return

            # Only store data for selected indices
            pred = pred[store_mask]
            target = target[store_mask]
            if mask is not None:
                mask = mask[store_mask]
            if meta is not None:
                meta = {k: v[store_mask] if isinstance(v, torch.Tensor) else v for k, v in meta.items()}

            self.batch_offsets[stage] = batch_end

            out = {'output': pred, 'target': target}
            if mask is not None:
                out['mask'] = mask
            if meta is not None:
                out['meta'] = meta

            self.data[stage].append(out)

    def _log_images(self, stage: str, trainer):
        if stage in self.stages and self.is_storing_on:
            data = self.data[stage].collate_results()

            if data['output'] is None:
                return

            output_imgs = torch.as_tensor(data['output'])
            target_imgs = torch.as_tensor(data['target'])

            images = self._make_combined_images(output_imgs, target_imgs)

            if 'mask' in data:
                mask_imgs = torch.as_tensor(data['mask'])
                masks = self._make_combined_masks(mask_imgs, mask_imgs)

                assert isinstance(trainer.logger, WandbLogger), 'Logger must be WanndB logger to log images with masks.'

                wandb_images = []
                for img, mask in zip(images, masks):
                    mask_dict = {'mask':{'mask_data':mask}}
                    if self.class_labels is not None:
                        mask_dict['mask']['class_labels'] = self.class_labels
                    wandb_images.append(wandb.Image(img, masks=mask_dict))
                metrics = {f"samples_{stage}": wandb_images}
                trainer.logger.log_metrics(metrics=metrics)

            else:
                trainer.logger.log_image(key=f"samples_{stage}", images=images) 

    def _make_combined_images(self, outputs: torch.Tensor, targets: torch.Tensor) -> list:
        images = []
        for o, t in zip(outputs, targets):
            img = make_grid([o, t], nrow=2, padding=0)
            img = to_pil_image(img)
            images.append(img)
        return images

    def _make_combined_masks(self, outputs: torch.Tensor, targets: torch.Tensor) -> list:
        masks = []
        for o, t in zip(outputs, targets):
            mask = torch.concat((o, t), dim=-1)
            mask = mask.numpy().squeeze()
            masks.append(mask)
        return masks

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_images('train', trainer)
        self._clear_output('train')
        self.batch_offsets['train'] = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_images('val', trainer)
        self._clear_output('val')
        self.batch_offsets['val'] = 0

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_images('test', trainer)
        self._clear_output('test')
        self.batch_offsets['test'] = 0


class VisualFieldViewer(GatherCallback):
    def __init__(self, stages, store_interval = 10, num_vf: int = 5, class_labels: dict[int, str] = None):
        super().__init__(stages, store_interval)

        self.num_vf = num_vf
        self.vf_indices = {}
        self.batch_offsets = {}

        # set static backend to generate png images from figures
        matplotlib.use('agg')

        if isinstance(class_labels, Mapping):
            self.class_labels = dict(class_labels)

    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)

        rng = np.random.default_rng(seed=42)
        for stage in self.stages:
            dataset_size = self._get_dataset_size(trainer=trainer, stage=stage)
            indices = np.arange(dataset_size)
            rng.shuffle(indices)
            selected_indices = indices[:self.num_vf] if self.num_vf < dataset_size else indices[:dataset_size]
            self.vf_indices[stage] = selected_indices

            self.batch_offsets[stage] = 0

    def _get_dataset_size(self, trainer, stage: str) -> int:
        """Get the number of samples in the dataset for a given stage."""
        if stage == 'train':
            return len(trainer.datamodule.train_dataloader().dataset)
        elif stage == 'val':
            return len(trainer.datamodule.val_dataloader().dataset)
        elif stage == 'test':
            return len(trainer.datamodule.test_dataloader().dataset)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def on_sanity_check_start(self, trainer, pl_module):
        """
        Define separate set of indices for sanity checking.
        Only subset of dataset is used, thus selecting indices based on the whole dataset would not work.
        """

        self.vf_indices['val_temp'] = self.vf_indices['val']

        num_sanity_batches = trainer.num_sanity_val_steps
        val_dataloader = trainer.datamodule.val_dataloader()
        batch_size = val_dataloader.batch_size

        dataset_size = batch_size * num_sanity_batches
        indices = np.arange(dataset_size)

        self.vf_indices['val'] = indices[:self.num_vf] if self.num_vf < dataset_size else indices[:dataset_size]

    def on_sanity_check_end(self, trainer, pl_module):
        self.vf_indices['val'] = self.vf_indices.pop('val_temp')

    def _store_output(self, stage: str, outputs):
        """
        Overwrite store function to only store the selected visual fields.
        Otherwise out of memory error can arise.
        """
        if stage in self.stages and self.is_storing_on:

            pred = outputs['output']
            target = outputs['target']
            mask = outputs.get('mask', None)
            meta = outputs.get('meta', None)

            batch_size = len(outputs['output'])

            # Get batch indices for this stage
            batch_start = self.batch_offsets[stage]
            batch_end = batch_start + batch_size
            batch_indices = np.arange(batch_start, batch_end)

            # Check which indices from this batch should be stored
            target_indices = self.vf_indices.get(stage, [])
            store_mask = np.isin(batch_indices, target_indices)

            if not store_mask.any():
                self.batch_offsets[stage] = batch_end
                return

            # Only store data for selected indices
            pred = pred[store_mask]
            target = target[store_mask]
            if mask is not None:
                mask = mask[store_mask]
            if meta is not None:
                meta = {k: v[store_mask] if isinstance(v, torch.Tensor) else v for k, v in meta.items()}

            self.batch_offsets[stage] = batch_end

            out = {'output': pred, 'target': target}
            if mask is not None:
                out['mask'] = mask
            if meta is not None:
                out['meta'] = meta

            self.data[stage].append(out)

    def _log_visual_fields(self, stage: str, trainer):
        if stage in self.stages and self.is_storing_on:
            data = self.data[stage].collate_results()

            if data['output'] is None:
                return

            output_vf = torch.as_tensor(data['output'])
            target_vf = torch.as_tensor(data['target'])
            coords = torch.as_tensor(data['meta.coordinates'])

            images = self._make_combined_vf(output_vf, target_vf, coords)

            trainer.logger.log_image(key=f"samples_{stage}", images=images) 

    def _make_combined_vf(self, outputs: torch.Tensor, targets: torch.Tensor, coordinates: torch.Tensor) -> list:
        vf_images = []
        for o, t, c in zip(outputs, targets, coordinates):
            x = c[:, 0]
            y = c[:, 1]
            vf_output = VisualField.from_arrays(x, y, o)
            vf_target = VisualField.from_arrays(x, y, t)
            
            # Create subplot with both VFs
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            vf_output.plot_voronoi(ax=ax1, show_lines=False, show_values=False)
            ax1.set_title('Predicted')
            
            vf_target.plot_voronoi(ax=ax2, show_lines=False, show_values=False)
            ax2.set_title('Target')
            
            plt.tight_layout()
            
            # Convert to PIL and add to list
            img = matplotlib_to_pil(fig)
            vf_images.append(img)
            
            plt.close(fig)
        return vf_images

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_visual_fields('train', trainer)
        self._clear_output('train')
        self.batch_offsets['train'] = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_visual_fields('val', trainer)
        self._clear_output('val')
        self.batch_offsets['val'] = 0

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_visual_fields('test', trainer)
        self._clear_output('test')
        self.batch_offsets['test'] = 0

def matplotlib_to_pil(fig) -> Image.Image:
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    return img.copy()  # Copy to avoid issues when buffer is closed


class GradCamCallback(Callback):

    def __init__(
        self,
        target_layers: list[str],
        store_interval: int = 0,
        save_to_dir=True,
        n_save=100,
        log_to_wandb=True,
        n_log=6,
    ):
        """
        Grad-CAM callback using the pytorch-grad-cam package.

        Args:
            target_layers (list[str]): Names of model modules to use as CAM target layers
            (must match keys from model.named_modules()). E.g. for a ResNet 'layer4.1' is a feasible target
            store_interval (int): Epoch interval at which CAMs are computed and stored.
            Set to 0 to store only the last validation epoch.
            save_to_dir (bool): If True, save generated CAM images to disk under
            trainer.default_root_dir / 'grad_cam'.
            n_save (int): Maximum number of images to save to disk.
            log_to_wandb (bool): If True, log images to Weights & Biases via trainer.logger.
            n_log (int): Maximum number of images to log to wandb.

        Notes:
            - The callback uses the provided target layer names to locate modules in the
              model via model.named_modules(). Missing layers will be skipped with a warning.
            - Validation in Lightning runs under torch.no_grad(); gradients are temporarily
              enabled when computing CAM masks.
        """
        super().__init__()

        # configurable fields
        self.target_layers = target_layers

        self.colormap = "jet"
        self.store_interval = store_interval

        self.log_to_wandb = log_to_wandb
        self.n_log: int = n_log
        self.log_count = 0

        self.save_to_dir = save_to_dir
        self.make_archive = True
        self.n_save: int = n_save
        self.save_count = 0

        self.cam = None
        self.aug_smooth = False
        self.eigen_smooth = False

        self.grad_cam_dir = None
        self.out_dir = None
        self.sample_idx = 0

        self.is_storing_on = False

        # set static backend to generate png images from figures
        matplotlib.use('agg')

    def setup(self, trainer, pl_module, stage):
        self.grad_cam_dir = Path(trainer.default_root_dir) / 'grad_cam'
        self.grad_cam_dir.mkdir()

    def _toggle_storing(self, pl_module):
        count = pl_module.current_epoch
        if self.store_interval == 0:
            self.is_storing_on = False
        elif count % self.store_interval == 0:
            self.is_storing_on = True
        else:
            self.is_storing_on = False

        # always store the last epoch
        if pl_module.current_epoch == pl_module.trainer.max_epochs - 1:
            self.is_storing_on = True  

    def _get_target_modules(self, pl_module):
        model = getattr(pl_module, "model", pl_module)
        modules = []
        for lname in self.target_layers:
            m = self._get_module_by_name(model, lname)
            if m is None:
                LOGGER.warning("GradCAM: target layer '%s' not found in model.", lname)
                continue
            modules.append(m)
        return modules

    def _get_module_by_name(self, model: torch.nn.Module, layer_name: str):
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None

    def on_validation_start(self, trainer, pl_module):
        self._toggle_storing(pl_module)

        if not self.is_storing_on:
            return

        model = getattr(pl_module, "model", pl_module)
        target_layer = self._get_target_modules(pl_module)
        self.cam = GradCAM(model=model, target_layers=target_layer)

        if self.save_to_dir:
            self.out_dir = self.grad_cam_dir / f'epoch_{pl_module.current_epoch}'
            self.out_dir.mkdir(exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        self.cam = None
        self.sample_idx = 0
        self.log_count = 0
        self.save_count = 0

        if self.is_storing_on and self.save_to_dir and self.make_archive:
            # archive the images to avoid blowing up the file count quota on Ubelix
            shutil.make_archive(self.out_dir, format='tar', root_dir=self.out_dir.parent, base_dir=self.out_dir.name)
            shutil.rmtree(self.out_dir)

        return super().on_validation_end(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):

        if not self.is_storing_on:
            return

        if not self._is_logging_on() and not self._is_saving_on():
            return 

        inputs, targets, meta, *_ = batch
        outputs = outputs['output']

        cam_masks = self.create_cam_masks(inputs)

        # Move data to cpu and prepare it for visualization
        inputs = inputs.detach().cpu().numpy()
        targets = targets.detach().cpu().squeeze().numpy()
        outputs = outputs.detach().cpu().squeeze().numpy()

        images_to_log = []
        sample_ids = self._get_sample_ids(meta)

        for img, cam, targ, pred, s_id in zip(inputs, cam_masks, targets, outputs, sample_ids):

            cam_img = self.create_cam_image(img, cam, pred, targ)

            if self._is_saving_on():
                cam_img.save(self.out_dir / f'{s_id}.png')
                self.save_count += 1

            if self._is_logging_on():
                images_to_log.append(cam_img)
                self.log_count += 1

        if self.log_to_wandb:
            # log once every batch to avoid too many wandb api calls
            trainer.logger.log_image(key=f"GradCam_validation", images=images_to_log) 

    def _is_logging_on(self) -> bool:
        return self.log_to_wandb and self.log_count <= self.n_log

    def _is_saving_on(self) -> bool:
        return self.save_to_dir and self.save_count <= self.n_save

    def _get_sample_ids(self, meta: dict) -> list[str]:

        heyex_id = meta.get('heyex_id_anon', None)
        laterality = meta.get('laterality', None)
        acquisition_date = meta.get('acquisition_date', None)
        id_cols = [heyex_id, laterality, acquisition_date]

        if all([x is not None for x in id_cols]):
            return [f'{h}_{l}_{d}' for h, l, d, in zip(*id_cols)]
        else:
            raise ValueError('Missing meta data to generate sample ids')

    def create_cam_masks(self, input_images: torch.Tensor) -> np.ndarray:
        # Validation in lightning happens in no_grad context, enable grads temporary for cam
        with torch.enable_grad():
            input_images = input_images.requires_grad_(True)
            cam_masks = self.cam(input_images, aug_smooth=self.aug_smooth, eigen_smooth=self.eigen_smooth)
            # cam removes channel dimension, add it to have same number of dimensions as input
            cam_masks = cam_masks[:,np.newaxis,]
            assert input_images.dim() == cam_masks.ndim, f'Dimension mismatch of input images of shape {input_images.shape} and cam masks of shape {cam_masks.shape}'
        return cam_masks

    def create_cam_image(self, img: np.ndarray, cam: np.ndarray, prediction = None, target = None) -> Image.Image:

        assert img.ndim == 3, 'Expected image of shape C,H,W'
        assert cam.ndim == 3, 'Expected cam mask of shape C,H,W'

        img = np.moveaxis(img, 0, -1)   # C,H,W -> H,W,C
        cam = np.moveaxis(cam, 0, -1)   # C,H,W -> H,W,C

        if img.shape[-1] > 1:
            img = img[..., 0] # select only the first channel of the image

        fig = plt.figure()
        ax = fig.subplots()
        ax.imshow(img, cmap='gray')
        ax.imshow(cam, cmap=self.colormap, alpha=0.3)

        # if (prediction is not None) and (target is not None):
        #     plt.title(f'target: {target:.2f}\nprediction: {prediction:.2f}')

        fig_img = matplotlib_to_pil(fig)
        plt.close(fig)

        return fig_img


class ArchiveCallback(Callback):
    def __init__(self):
        super().__init__()

    def teardown(self, trainer, pl_module, stage):
        return super().teardown(trainer, pl_module, stage)


class VisualFieldEvalCallback(GatherCallback):
    "Callback for custom evaluations of on the predicted visual fields at the end of the training"
    def __init__(self, run_dir: str):
        super().__init__(stages=['val'], store_interval=-1, keys_to_store=['output', 'target', 'meta'])

        self.output_dir = Path(run_dir) / 'output'

    def _eval_results(self, stage: str, trainer, pl_module ):
       
        data = self.data[stage].collate_results()


        vf_target = data['target']
        vf_output = data['output']

        # average over vf dimension, works also for single value md predictions
        md_target = np.mean(vf_target, axis=1)
        md_output = np.mean(vf_output, axis=1)

        stage_target = [str(GlaucomaStage.from_mean_deviation_octopus(x)) for x in md_target]

        results = pd.DataFrame.from_dict({'md_target': md_target, 'md_output':md_output, 'stage': stage_target})
        results['error'] = results['md_output'] - results['md_target']


        r2_scorer = R2Score()
        mae_scorer = MeanAbsoluteError()

        metrics = []
        for stage in np.append(results['stage'].unique(), 'all'):

            if stage == 'all':
                stage_mask =  np.full_like(results['stage'], True)
            else:
                stage_mask = results['stage'] == stage

            stage_target = torch.tensor(results.loc[stage_mask, 'md_target'].values, dtype=torch.float32)
            stage_output = torch.tensor(results.loc[stage_mask, 'md_output'].values, dtype=torch.float32)
            
            errors = stage_output - stage_target
            count = len(errors)
            me = errors.mean()
            std = errors.std()
            
            if count > 2:
                r2 = r2_scorer(stage_output, stage_target)
            else:
                r2 = np.nan

            mae = mae_scorer(stage_output, stage_target)
            metrics.append([stage, count, mae, me, std, r2])

        metrics_df = pd.DataFrame.from_records(metrics, columns=['stage','count', 'mean abs error', 'mean error','std','R2', ])
        metrics_df.sort_values('stage', inplace=True)

        trainer.logger.log_table(key="final/val_metrics", dataframe = metrics_df)
        

    def on_validation_epoch_end(self, trainer, pl_module):
        if 'val' in self.stages and self.is_storing_on:
            self._eval_results('val', trainer, pl_module)
            self._clear_output('val')
    
    def on_test_epoch_end(self, trainer, pl_module):
        self._eval_results('test', trainer, pl_module)
        self._clear_output('test')
