from pathlib import Path
import urllib
import subprocess

from omegaconf import OmegaConf
import wandb

from eyemod.run_utils import _get_wandb_run_dir, _get_wandb_run_id


class WandBRun:
    """
    Run Object that wraps a wandb.Run object and provides access to associated
    local data (run directory, checkpoints, etc.) that may not be uploaded to WandB.
    """
    
    def __init__(self, id: str, project: str, entity: str, config: dict = None, path: str | Path = None):
        
        self.id = id
        self.project = project
        self.entity = entity

  
        self._config = OmegaConf.create(config) if config else None
        self._path = Path(path) if path else None
        self._host = None
        
    @property
    def wandb_path(self):
        return f'{self.entity}/{self.project}/{self.id}'
    
    @property
    def config(self):
        if self._config is None:
            if self._path is not None:
                self._config = self._get_config_from_local()
            else:
                try:
                    self._config = self._get_config_from_online()
                except:
                    raise ValueError("Configuration is not available.")
        return self._config

        
    @property
    def path(self):
        if self._path is None:
            try:
                self._path = Path(self.config.dirs.run)
            except:
                raise ValueError("Path is not available. It can not be extracted from config.")
        return self._path

    @property
    def host(self):
        if self._host is None:
            device = self.config.env.device
            if device == 'ubelix':
                self._host = 'submit01.unibe.ch'
            else:
                self._host = 'localhost'
            return self.host
        else: 
            return self._host
    
    def _get_config_from_local(self):
        return OmegaConf.load(self.path / 'config.yaml')

    def _get_config_from_online(self):
        run = self._get_online_run()
        return OmegaConf.create(run.config)
    
    def _get_online_run(self):
        try:
            return wandb.Api().run(self.wandb_path)
        except Exception as e:
            raise ValueError(f"Could not find run {self.wandb_path} online") from e

    def exists_online(self) -> bool:
        try:
            run = wandb.Api().run(self.wandb_path)
            return True
        except:
            return False
    
    def exists_local(self) -> bool:
        try:
            return self.path.exists()
        except:
            return False

    @classmethod
    def from_wandb_run(cls, run) -> 'WandBRun':
         return cls(id=run.id, project=run.project, entity=run.entity, config=run.config)

    @classmethod
    def from_wandb_path(cls, wandb_path: str) -> 'WandBRun':
        run = wandb.Api().run(wandb_path)
        return cls.from_wandb_run(run)
    
    @classmethod
    def from_url(cls, url: str) -> 'WandBRun':
        path = urllib.parse.urlparse(url).path
        # decode url e.g. to replace %20 with spaces
        path = urllib.parse.unquote(path)
        # url contains "runs/" expression between project and run_id
        path = path.replace('runs/', '')
        return cls.from_wandb_path(path)
    
    @classmethod
    def from_run_dir(cls, path: str | Path)  -> 'WandBRun':
        path = Path(path)
        config = OmegaConf.load(path / 'config.yaml')
        logging = config.get('logging')
        wandb_dir = _get_wandb_run_dir(path)
        run_id = _get_wandb_run_id(wandb_dir)
        return cls(id=run_id, project=logging.project, entity=logging.entity, config=config, path=path)


def download_run(run: WandBRun, out_dir: Path, verbose: bool = True) -> Path:
    """
    Make sure you have configured your ssh connection to the host, to use this download function 
    """
    
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / run.path.name
    if run.path.exists():
        print('run exists on machine -> create symlink')
        out_path.symlink_to(run.path)

    elif out_path.exists():
        print('run already downloaded')

    else:
        print(f"Downloading {run.id} from {run.host}...")
        remote_path = f'{run.host}:"{run.path}"'
        result = subprocess.run(
            ['rsync', '-avz', '--progress', str(remote_path), str(out_path.parent)],
            capture_output= not verbose,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download from SLURM: {result.stderr}")
        print(f"Finished downloading {run.id}")
    return out_path
        




