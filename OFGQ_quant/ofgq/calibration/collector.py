

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch


class FormatCollector:


    def __init__(self):
        self._data = defaultdict(lambda: defaultdict(list))

    def add(
        self,
        key: str,
        layer_idx: int,
        indices: torch.Tensor,
    ) -> None:
        self._data[key][int(layer_idx)].append(
            indices.detach().cpu().to(torch.int64)
        )

    @property
    def data(self):
        return self._data

    def keys(self):
        return self._data.keys()
