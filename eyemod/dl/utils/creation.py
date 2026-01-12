from pathlib import Path
from datetime import datetime

import hydra
import wandb

from torch import nn
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf, ListConfig
from lightning import LightningDataModule, LightningModule, Callback
from lightning.pytorch.loggers import WandbLogger

from eyemod.dl.utils.datamodule_wrapper import DataModuleWrapper
from eyemod.config_helper import extract_target_name

def create_directories(config: DictConfig) -> None:
    assert "dirs" in config, "Dirs must be defined in the config"
    assert "procedure" in config, "Procedure must be defined in the config"
    assert "model" in config, "Model must be defined in the config"
    assert "dataset" in config, "Dataset must be defined in the config"
    assert "project" in config.logging, "Logging must have a project name"

    timestamp = datetime.now().strftime("%y%m%d-%H-%M-%S")

    job_id = OmegaConf.select(config, "env.slurm.job_id", default=None)
    job_id = job_id if job_id else "000000"

    project_name = config.logging.project
    procedure_name = extract_target_name(config.procedure)
    model_name = extract_target_name(config.model)
    dataset_name = extract_target_name(config.dataset.train)

    run_name = f"{job_id}_{timestamp}_{dataset_name}"

    log_dir = Path(config.dirs.logs)
    run_dir = log_dir / project_name /procedure_name / model_name / run_name
    ckpt_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save the new directories in the config
    config.dirs["run"] = str(run_dir)
    config.dirs["checkpoints"] = str(ckpt_dir)
    
    return config


def create_logger(config: DictConfig) -> WandbLogger:
    """
    Creates and initiates a WandB logger instance.
    Information is logged to wandb starting after the init call. 
    """
    assert 'logging' in config, "'logging' must be defined in the config"
    assert 'run' in config.dirs, "'dirs.run' must be defined in the config"

    wandb_run = wandb.init(
        **config.logging,
        dir=config.dirs.run,
        config=OmegaConf.to_object(config),
    )
    return WandbLogger(experiment=wandb_run)

def create_datamodule(config: DictConfig) -> LightningDataModule:
    """
    Creates an datamodule instance based on the config.
    Creates the transforms, datasets and dataloaders for each split and warps them in a Datamodule.
    """
    assert "transformations" in config, "Transformations must be defined in the config"
    assert "target_transformations" in config, "Target_transformations must be defined in the config"
    assert "dataset" in config, "Dataset must be defined in the config"
    assert "dataloader" in config, "Dataloader must be defined in the config"

    dataloaders = {}
    for split in config.dataset:
        
        dataset = create_dataset(config, split)

        if "batch_sampler" in config.dataloader[split]:
            dataloader = instantiate(
                config.dataloader[split], dataset=dataset
            )
        else:
            dataloader = instantiate(config.dataloader[split], dataset=dataset)

        dataloaders[split] = dataloader

    return DataModuleWrapper.fromdict(dataloaders)

def create_dataset(config: DictConfig, split: str) -> Dataset:

    assert 'dataset' in config, "'dataset' must be defined in the config"
    assert 'transformations' in config, "'transformations' must be defined in the config"
    assert 'target_transformations' in config, "'target_transformations' must be defined in the config"


    transform = create_transformation(config.transformations[split])
    target_transform = create_transformation(config.target_transformations[split])

    dataset = instantiate(config.dataset[split], transform=transform, target_transform=target_transform)
    return dataset


def create_transformation(config: DictConfig) -> nn.Module:
    if config is None:
        return None
    else:
        if isinstance(config, ListConfig):
            transforms = [instantiate(c) for c in config]
            assert all([isinstance(t, nn.Module) for t in transforms]), 'All transformations must be nn.Modules'
            return transforms
        elif isinstance(config, DictConfig):
            transform = instantiate(config)
            assert isinstance(transform, nn.Module), 'Transformation must be a nn.Module'
            return transform
        else:
            raise ValueError('Cannot create transformation, Invalid config')


def create_procedure(config: DictConfig) -> LightningModule:

    assert 'procedure' in config, "'procedure' must be defined in the config"
    assert 'optimizer' in config, "'optimizer' must be defined in the config"

    kwargs = {}
    kwargs['optimizer'] = instantiate_partial(config.optimizer)
    if 'scheduler' in config:
        kwargs['scheduler'] = instantiate_partial(config.scheduler)

    return instantiate(config.procedure, **kwargs)

def create_callbacks(config: DictConfig) -> list[Callback]:
    assert 'callbacks' in config, "'callbacks' must be defined in the config"
    return instantiate(config.callbacks)

def instantiate(config: DictConfig, **kwargs):
    """
    Creates an instance of an object defined by a dot path stored under '_target_'.
    
    Future: Get rid of hydra implementation to reduce dependencies.
    """
    return hydra.utils.instantiate(config, **kwargs)

def instantiate_partial(config: DictConfig, **kwargs):
    config = config.copy()
    config['_partial_'] = True
    return hydra.utils.instantiate(config, **kwargs)

if __name__ == '__main__':
    
    config = DictConfig({'_target_':'torch.optim.AdamW',
                         'lr':0.001})
    print(config)

    optim = instantiate_partial(config=config)
    print(type(optim))
    print(config)

