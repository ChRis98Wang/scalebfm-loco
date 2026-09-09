import os
import glob
from enum import Enum
from pathlib import Path
from omegaconf import DictConfig
from loguru import logger

class Mode(Enum):
    file = 0
    dir = 1


def load_motion_manifest(data_path, manifest_path, *, allowed_suffixes):
    """Resolve an exact, traversal-safe motion list below a prepared data root."""
    data_root = Path(data_path).resolve()
    manifest = Path(manifest_path)
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"Motion manifest is not a regular file: {manifest}")
    selected = []
    seen = set()
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line:
            raise ValueError(f"Motion manifest contains an empty entry: {manifest}")
        relative_path = Path(raw_line)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Motion manifest entry escapes the data root: {relative_path}"
            )
        motion_path = data_root / relative_path
        if motion_path in seen:
            raise ValueError(f"Duplicate motion manifest entry: {relative_path}")
        if not motion_path.is_file() or motion_path.suffix not in allowed_suffixes:
            raise ValueError(f"Invalid motion manifest entry: {motion_path}")
        seen.add(motion_path)
        selected.append(str(motion_path))
    if not selected:
        raise ValueError(f"Motion manifest is empty: {manifest}")
    return selected


class BaseLoader:
    def __init__(self, config: DictConfig):
        self.config = config
        self.format = self.config.data_format
        self.overwrite = self.config.get('overwrite', False)
        self.target_fps = self.config.get('target_fps', 30)
        self.output_dir = os.path.expanduser(self.config.get('output_dir', 'retargeted_dataset'))
        self.skipped_num = 0
        self.failed_num = 0
        self.failure_messages = []
        
    def load(self, data_path):
        self.data_path = data_path

        if os.path.isdir(self.data_path):
            self.mode = Mode.dir
            self.data_list = glob.glob(f"{self.data_path}/**/*{self.format}", recursive=True)
            self.data_num = len(self.data_list)
        elif os.path.isfile(self.data_path):
            suffix = os.path.splitext(self.data_path)[-1]
            assert suffix == self.format, f"You are using the data loader for {self.format} format but the data you give is {suffix} format!"
            self.mode = Mode.file
            self.data_list = [self.data_path]
            self.data_num = 1
        else:
            raise Exception(f"Check your data path: {self.data_path}")
        
        logger.info(f"[Loader] Total number of motions: {self.data_num}")
        
    def __iter__(self):
        self.current_idx = 0
        self.skipped_num = 0
        self.failed_num = 0
        self.failure_messages = []
        
        while self.current_idx < self.data_num:
            try: 
                sample_path = self.data_list[self.current_idx]
                if self.mode == Mode.file:
                    relative_path = os.path.basename(sample_path)
                else:
                    data_dir_name = Path(self.data_path).resolve().name
                    relative_path = os.path.join(
                        data_dir_name,
                        os.path.relpath(sample_path, self.data_path),
                    )
                relative_path = os.path.splitext(relative_path)[0] + '.pkl'
                save_path = os.path.join(self.output_dir, relative_path)

                if os.path.abspath(save_path) == os.path.abspath(sample_path):
                    raise ValueError(
                        f"Output path resolves to the source file: {sample_path}. "
                        "Choose a different output_dir."
                    )
                if os.path.exists(save_path) and not self.overwrite:
                    logger.warning(f"[Loader] {save_path} already exists!")
                    self.skipped_num += 1
                    self.current_idx += 1
                    continue
                frames, extras = self._load_sample(sample_path)
                yield save_path, frames, extras
                self.current_idx += 1
            except Exception as e:
                logger.warning(f"[Loader] File {self.data_list[self.current_idx]} is broken! Skip it for error {e}")
                self.failed_num += 1
                self.failure_messages.append(
                    f"{self.data_list[self.current_idx]}: {e}"
                )
                self.current_idx += 1

    def _load_sample(self, sample_path: Path):
        raise NotImplementedError
