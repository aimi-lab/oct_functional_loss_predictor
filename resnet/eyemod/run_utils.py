import re
from pathlib import Path
from typing import Protocol, Any
import shutil
import json
import yaml

from abc import ABC, abstractmethod

import wandb

class WandbRunProtocol(Protocol):
    id: str
    project: str


class LocalWandbRun(ABC):
    """Abstract base class for local wandb run implementations."""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.project = None
        self.id = None
    
    @abstractmethod
    def delete(self):
        """Delete the local run data."""
        pass
 
class LocalWandbRunDir(LocalWandbRun):
    """Local implementation of wandb run that mimics the wandb.apis.public.Run interface."""

    def __init__(self, path: Path):
        super().__init__(path)
        assert path.is_dir(), f"Expected {path} to be a directory"
        self.wandb_run_dir = _get_wandb_run_dir(self.path)

        if self.wandb_run_dir is None:
            raise ValueError(f"No wandb run directory found in {path}. This was not a wandb run.")

        self.project = self._get_project_name(self.path)
        self.id = _get_wandb_run_id(self.wandb_run_dir)
    
    def _get_project_name(self, run_dir: Path) -> str:
        # config_file = run_dir / 'config.yaml'
        config_files = list(run_dir.rglob('config.yaml'))
        if len(config_files) == 0:
            raise FileNotFoundError(f'Could not find config file of run stored at {self.path}')
        else:
            # folder can have multiple copies of the config file
            config_file = config_files[0]

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        project_name = find_value_by_key(config, 'project')
        if project_name is None:
            raise ValueError(f'Could not find project key in config file for run stored at {self.path}')
        return  project_name


    def _load_metadata(self, wandb_run_dir) -> dict:
        """Load metadata from the local wandb directory structure."""
        meta_file = wandb_run_dir / 'files' / 'wandb-metadata.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            return  metadata
        else:
            raise FileNotFoundError(f'Could not find meta data file of run {self.id} stored at {self.path}')
        
    def delete(self):
        shutil.rmtree(self.path)

    def archive(self, compress: bool = True, output_dir: Path = None) -> Path:
        """Create an archive from the run directory."""
        run_id = generate_run_identifier(self)
        
        # Set default output directory to parent of run directory
        if output_dir is None:
            output_dir = self.path.parent
        
        # Create archive filename
        archive_name = output_dir / f"{self.path.name}_{run_id}"
        archive_format = 'gztar' if compress else 'tar'
        
        # Create the archive
        archive_path = shutil.make_archive(
            base_name=str(archive_name),
            format=archive_format,
            root_dir=self.path.parent,
            base_dir=self.path.name
        )

        if not Path(archive_path):
            raise FileNotFoundError(f"Archive creation failed: {archive_path}")
        
        return Path(archive_path)
    
def _get_wandb_run_dir(run_dir: Path) -> Path:
    wandb_dir = run_dir / "wandb"
    # if the run was logged with wandb, the wandb directory exists
    if wandb_dir.exists():
        sub_dirs = list(wandb_dir.glob("*run-*"))
        assert (
            len(sub_dirs) == 1
        ), f"Expected exactly one subdirectory containing 'run-' in its name in {wandb_dir}"
        # offline runs have a prefix "offline-" in the directory name
        return sub_dirs[0]
    else:
        return None

def _get_wandb_run_id(wandb_run_dir: Path) ->str:
    # name pattern of dir is:
    # run-<time_stamp>-<id> or offline-run-<time_stamp>-<id>
    dir_name = wandb_run_dir.name
    dir_name = dir_name.removeprefix("offline-")
    _, wandb_id = dir_name.split("-")[-2:]
    return wandb_id

class LocalWandbRunArchive(LocalWandbRun):
    def __init__(self, path: Path):
        super().__init__(path)
        assert path.is_file(), f"Expected {path} to be a file"
        project, run_id = self._get_run_info(self.path.name)
        self.project = project
        self.id = run_id

    def _get_run_info(self, archive_name: str):
        base_name = archive_name.split('.')[0]
        run_identifier = base_name.split('_run_')[-1]
        project, run_id = run_identifier.split('_')
        return project, run_id
    
    def delete(self):
        self.path.unlink()


def find_value_by_key(d, target_key):
    """Extract value of a key at any level in nested dict/list structure."""
    if isinstance(d, dict):
        if target_key in d:
            return d[target_key]
        for value in d.values():
            result = find_value_by_key(value, target_key)
            if result is not None:
                return result
    elif isinstance(d, list):
        for item in d:
            result = find_value_by_key(item, target_key)
            if result is not None:
                return result
    return None


def generate_run_identifier(run: WandbRunProtocol) -> str:

    if run.id is None:
        raise ValueError(f"Run is missing wandb id")
    if run.project is None:
        raise ValueError(f"Run is missing the project name")
    project = run.project.replace(' ', '').replace('_', '').lower()
    return f'run_{project}_{run.id}'


def collect_local_runs(log_dir: Path) -> list[LocalWandbRun]:
    dir_paths = get_local_run_dirs(log_dir = log_dir)
    run_dirs = []
    for p in dir_paths:
        try:
            run_dirs.append(LocalWandbRunDir(p))
        except Exception as e:
            print(e)
            continue

    archive_paths = get_local_run_archives(log_dir=log_dir)
    run_archives = []
    for p in archive_paths:
        try:
            run_archives.append(LocalWandbRunArchive(p))
        except Exception as e:
            print(e)
            continue

    run_dirs.extend(run_archives)
    return run_dirs

def get_local_run_dirs(
    log_dir: Path, pattern: str = r"(0{6}|\d{7,8})_\d{6}-\d{2}-\d{2}-\d{2}_"
) -> list[Path]:
    """
    Get all run directories found in the provided log directory and its subdirectories.

    default pattern to match:
    for local runs: matches 6 zeros followed by a timestamp e.g. 000000_210101-12-00-00_<run_name>
    for slurm runs: matches 7 or 8 digits (job id) followed by a timestamp e.g. 1234567_210101-12-00-00_<run_name>
    """
    pattern = re.compile(pattern)
    
    # Use glob patterns for faster filesystem traversal when possible
    # This avoids checking every single directory/file
    try:
        # Try direct glob patterns first for common cases
        if pattern.pattern == r"(0{6}|\d{7,8})_\d{6}-\d{2}-\d{2}-\d{2}_":

            run_dirs = []
            # Use multiple glob patterns for the expected run dir cases
            run_dirs.extend(log_dir.rglob("000000_*/"))  # Local runs
            run_dirs.extend(log_dir.rglob("???????_*/"))  # 7-digit SLURM
            run_dirs.extend(log_dir.rglob("????????_*/"))  # 8-digit SLURM
            # Filter to only directories and validate with regex
            run_dirs = [d for d in run_dirs if d.is_dir() and pattern.match(d.name)]
        else:
            # Fallback to original approach for custom patterns
            run_dirs = [d for d in log_dir.rglob("**/") if d.is_dir() and pattern.match(d.name)]
    except Exception:
        # Fallback to original approach if glob patterns fail
        run_dirs = [d for d in log_dir.rglob("**/") if d.is_dir() and pattern.match(d.name)]
    
    return run_dirs

def get_local_run_archives(log_dir: Path, pattern: str = r"(0{6}|\d{7,8})_\d{6}-\d{2}-\d{2}-\d{2}_.*_run_.*_[a-zA-Z0-9]+\.tar(\.gz)?$"
) -> list[Path]:
    """
    Get all run archives found in the provided log directory and its subdirectories.

    default pattern to match:
    for local runs: matches 6 zeros followed by a timestamp e.g. 000000_210101-12-00-00_<run_name>_run_<project>_<wandb_id>.tar(.gz)
    for slurm runs: matches 7 or 8 digits (job id) followed by a timestamp e.g. 1234567_210101-12-00-00_<run_name>_run_<project>_<wandb_id>.tar(.gz)
    """
    pattern = re.compile(pattern)
    archive_paths = [f for f in log_dir.rglob("*.tar*") if f.is_file() and pattern.match(f.name)]
    return archive_paths

