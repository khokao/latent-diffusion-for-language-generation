"""
The codes are modified.
Link:
    - [Meter] https://github.com/Megvii-BaseDetection/YOLOX/
      blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/utils/metric.py#L54-L98
"""
from collections import UserDict, deque

import numpy as np
import torch


class DictMeter(UserDict):
    def __init__(self, meter_dict):
        super().__init__(meter_dict)

    def update_all(self, values_dict):
        for k, v in values_dict.items():
            self[k].update(v)

    def reset_all(self):
        for m in self.values():
            m.reset()


class Meter:
    def __init__(self):
        self._deque = deque()
        self._count = 0
        self._total = 0.0

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.mean().item()

        self._deque.append(value)
        self._count += 1
        self._total += value

    def reset(self):
        self._deque.clear()
        self._count = 0
        self._total = 0.0

    @property
    def avg(self):
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None
