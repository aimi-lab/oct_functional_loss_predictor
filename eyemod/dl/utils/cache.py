from pathlib import Path
from typing import Iterable
from importlib import import_module
import shutil
import logging

from omegaconf import DictConfig
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def copy_dataset_to_local(config: DictConfig, cache_dir: Path):
    """
    Copy dataset files required by config to a local temp cache and return updated config.
    - cache_dir: path to use for cache (default $TMPDIR or tempfile.gettempdir()).
    - strategy: 'auto' (heuristics), 'full' (copy whole root), or 'selective' (use discovered files).
    This will update dataset root entries in the provided config (in-place).
    """

    cache_dir = Path(cache_dir)

    # iterate dataset entries in config.dataset (matches create_datamodule)
    dataset_configs = list(config.dataset.items())

    paths = set()
    for split_name, ds_config in dataset_configs:
        ds_files = get_required_data_of_dataset(ds_config)
        paths.update(ds_files)

    files = _unpack_dirs(paths)
    if len(files) == 0:
        LOGGER.warning(
            'No files copied to local node. ' \
            'Check if dataset class implements "files_to_cache" function'
        )
        return config

    dataset_dir_old = Path(config.dirs.datasets)

    print(f'Copy files to {cache_dir}')
    for f in tqdm(files, desc=f'Copying...', total=len(files)):

        rel_path = f.relative_to(dataset_dir_old)

        new_path = cache_dir / rel_path    
        new_path.parent.mkdir(parents = True, exist_ok = True)

        shutil.copy(f, new_path, follow_symlinks=True)

    config.dirs.datasets = str(cache_dir)

    return config

def get_required_data_of_dataset(dataset_config: DictConfig) -> list[Path]:

    target = dataset_config.get("_target_", None)
    if not target:
        return None
    try:
        module_path, class_name = target.rsplit(".", 1)
        mod = import_module(module_path)
        cls = getattr(mod, class_name)
        if hasattr(cls, "required_data"):
            return cls.required_data(dataset_config)
    except Exception:
        LOGGER.exception(f"Failed to call required_data for {target}")
    return None

def _unpack_dirs(paths: Iterable[Path]) -> set[Path]:
    paths = set(paths)
    for p in list(paths):
        if p.is_dir():
            paths.discard(p)
            for child in p.rglob('*'):
                if child.is_file():
                    paths.add(child)
    return paths


if __name__ == '__main__':

    from eyemod.conf import project
    from eyemod import config_helper
    from omegaconf import OmegaConf

    config_folder = project.CONFIG_DIR
    config_path = Path('conf/experiments/Resnet18_SingleTimepoint_onh_oct.yaml')

    config = OmegaConf.load(config_path)
    config = config_helper.expand_defaults(config, config_folder)

    config = config_helper.select_dataset_directory(config)

    config = config_helper.process_execution_stages(config)

    copy_dataset_to_local(config, 'out/temp_out')
