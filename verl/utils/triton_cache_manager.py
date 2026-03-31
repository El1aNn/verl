import json
import os
import uuid
from typing import Dict, Optional

from triton.runtime.cache import FileCacheManager


class LenientFileCacheManager(FileCacheManager):
    """Tolerate transiently corrupt Triton cache metadata files.

    In multi-process compile scenarios Triton's group metadata file may be found
    in an invalid/empty state. Treat that as a cache miss so the kernel can be
    recompiled instead of crashing the training job.
    """

    def _drop_corrupt_file(self, filepath: str) -> None:
        try:
            if os.path.exists(filepath):
                corrupt_path = f"{filepath}.corrupt.{os.getpid()}.{uuid.uuid4().hex}"
                os.replace(filepath, corrupt_path)
        except OSError:
            pass

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        grp_filename = f"__grp__{filename}"
        if not self.has_file(grp_filename):
            return None

        grp_filepath = self._make_path(grp_filename)
        try:
            with open(grp_filepath) as f:
                grp_data = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            self._drop_corrupt_file(grp_filepath)
            return None

        child_paths = grp_data.get("child_paths", None)
        if not isinstance(child_paths, dict):
            self._drop_corrupt_file(grp_filepath)
            return None

        result: Dict[str, str] = {}
        for child_name, child_path in child_paths.items():
            if isinstance(child_name, str) and isinstance(child_path, str) and os.path.exists(child_path):
                result[child_name] = child_path
        return result

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)

        assert self.lock_path is not None
        filepath = self._make_path(filename)
        rnd_id = str(uuid.uuid4())
        pid = os.getpid()
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, filepath)
        os.removedirs(temp_dir)
        return filepath
