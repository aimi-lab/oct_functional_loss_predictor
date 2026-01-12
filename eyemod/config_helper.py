from pathlib import Path
from typing import MutableMapping, MutableSequence

from omegaconf import DictConfig, OmegaConf


def select_dataset_directory(config: DictConfig) -> DictConfig:
    """
    Selects the correct dataset directory based on the dataset name.

    The user config can hold multiple dataset dirs, this function selects the correct one based on the dataset name.
    """
    assert "dirs" in config, "Dirs must be defined in the config"
    assert "dataset" in config, "Dataset must be defined in the config"
    
    dataset_name = extract_target_name(config.dataset.train)

    if isinstance(config.dirs.datasets, MutableSequence):
        key = list(config.dataset.keys())[0]
        entry = config.dataset[key]
        assert "root" in entry, "Root must be defined in the dataset entry"

        rel_dataset_dir = entry.root.split("]")[-1]
        rel_dataset_dir = rel_dataset_dir.lstrip('/')
        
        for dataset_dir in config.dirs.datasets:
            candidate_dir = Path(dataset_dir) / rel_dataset_dir
            if candidate_dir.exists():
                config.dirs.datasets = dataset_dir
                return config
        raise ValueError(f"Dataset directory not found for dataset {dataset_name}")
    else:
        return config
 

def extract_target_name(config: DictConfig) -> str:
    if '_target_' in config:
        return config._target_.split(".")[-1]
    else:
        return None
    
def process_execution_stages(config: DictConfig) -> DictConfig:
    stages = config.get('stages', None)
    if stages is None:
        return config
    
    # remove unused dataset stage
    to_remove = [s for s in config.dataset if s not in stages]
    [config.dataset.pop(s) for s in to_remove]

    return config  

def merge_overrides(config: DictConfig, overrides: list) -> DictConfig:
    pass

def expand_defaults(config: DictConfig, config_dir:Path) -> DictConfig:
    """
    Expands the default configurations within a given configuration dictionary.

    This function recursively searches for a special key `_default_` within the 
    configuration dictionary. If found, it resolves the default configuration 
    and merges it with the current configuration.

    The default can either be in dot notation or a file path. E.g.
        _default_: folder.filename
        _default_: folder/filename.yaml


    The default config entries are overwritten if the provided config contains the same entry.

    Args:
        config (DictConfig): The configuration dictionary to expand.
        config_dir (Path): The directory path where the configuration files are located.

    Returns:
        DictConfig: The expanded configuration dictionary.
    """
    DEF_KEY = '_default_'
    assert isinstance(config, (dict, DictConfig)), "Config must be of type dict or DictConfig"
    if isinstance(config, DictConfig):
        # convert o plain dict
        config = config._value()
    config = _recursively_expand(config, config_dir, DEF_KEY)
    return OmegaConf.create(config)

def _recursively_expand(config: dict, config_dir: Path, key: str):
    """
    Recursive expansion of plain dict with value interpolation limited to the value to expand.
    This is required to avoid interpolation using an incomplete config dict.
    """
    if isinstance(config, MutableMapping):
        # value interpolation is required for expanding dynamic default targets
        # (re-)create config dict to enable value interpolation  
        config = OmegaConf.create(config)
 
        while key in config:
            def_= _load(config.pop(key), config_dir)
            config = OmegaConf.merge(def_, config)

        config = OmegaConf.to_container(config, resolve=False)
        for k, v in config.items():
            config[k] = _recursively_expand(v, config_dir, key=key)

        return config
    else:
        return config

def _load(config_path: str, config_dir: Path):
    """
    Loads a yaml configuration file. 

    Args:
        config_path (str): The relative path of the configuration file. The name can include dots (.) which will be replaced by slashes (/).
        config_dir (Path): The directory where the configuration files are stored.

    Returns:
        DictConfig: The loaded configuration.

    Raises:
        AssertionError: If `config_path` is not a string.
        AssertionError: If the configuration file does not exist.
    """
    assert isinstance(config_path, str), "Config path must be a string."

    config_path = config_path.replace(".", "/")

    if not config_path.endswith('yaml'):
        config_path = config_path + ".yaml"

    config_path = config_dir / config_path
    assert config_path.exists(), f"Unable to load config. File {config_path} does not exist."

    return OmegaConf.load(config_path)


if __name__ == '__main__':

    config_dir = Path('/Users/moritzschmid/Code/eyemod/bin/conf')

    config = OmegaConf.load(config_dir / 'supervised_test_conf.yaml')
    config = expand_defaults(config, config_dir)
    print(config)
