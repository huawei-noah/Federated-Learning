import csv
import json
import os
from datetime import datetime
from typing import Optional

import torch


class Saver:
    def __init__(self, path: str, name: Optional[str] = None, overwrite: bool = True):
        self.path = path
        self.overwrite = overwrite
        self.name = (
            datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f") if name is None else name
        )
        self.full_path = os.path.join(self.path, self.name)
        self.csv_header = False
        os.makedirs(self.full_path, exist_ok=True)

    def _write_header(self, fields):
        with open(os.path.join(self.full_path, "log.csv"), "w") as f:
            csv.writer(f).writerow(fields.keys())
        self.csv_header = True

    def write_csv(self, **kwargs):
        if not self.csv_header:
            self._write_header(kwargs)
        with open(os.path.join(self.full_path, "log.csv"), "a") as f:
            csv.writer(f).writerow(kwargs.values())

    def write_meta(self, **kwargs):
        file_path = os.path.join(self.full_path, "meta.json")
        if not os.path.exists(file_path) or self.overwrite:
            with open(file_path, "w") as f:
                json.dump(kwargs, f)

    def cast_values(**kwargs):
        raise NotImplementedError

    def save_model(self, model):
        file_path = os.path.join(self.full_path, "model.pth")
        torch.save(model.state_dict(), file_path)
