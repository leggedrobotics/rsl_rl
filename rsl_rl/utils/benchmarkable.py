import numpy as np
import time
from typing import Callable, Dict


class Benchmark:
    def __init__(self):
        self.reset()

    def __call__(self):
        if self.running:
            self.end()
        else:
            self.start()

    def end(self):
        now = time.process_time_ns()

        assert self.running

        difference = now - self._timer
        self._timings.append(difference)

        self._timer = None

    def reset(self):
        self._timer = None
        self._timings = []

    @property
    def running(self):
        return self._timer is not None

    def start(self):
        self._timer = time.process_time_ns()

    @property
    def timings(self):
        return self._timings


class Benchmarkable:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._benchmark = False
        self._bm_data = dict()
        self._bm_fusions = []

    def _bm(self, name: str) -> None:
        if not self._benchmark:
            return

        if name not in self._bm_data:
            self._bm_data[name] = Benchmark()

        self._bm_data[name]()

    def _bm_flush(self) -> None:
        # TODO: implement
        for val in self._bm_data.values():
            val.reset()

        for fusion in self._bm_fusions:
            fusion["target"]._bm_flush()

    def _bm_fuse(self, target, prefix="") -> None:
        assert isinstance(target, Benchmarkable)
        assert target not in self._bm_fusions

        target._bm_toggle(self._benchmark)
        self._bm_fusions.append(dict(target=target, prefix=prefix))

    def _bm_report(self) -> Dict:
        data = dict()

        if not self._benchmark:
            return data

        for key, val in self._bm_data.items():
            data[key] = (np.mean(val.timings), len(val.timings))

        for fusion in self._bm_fusions:
            target = fusion["target"]
            prefix = fusion["prefix"]

            for key, val in target._bm_report().items():
                data[f"{prefix}{key}"] = val

        return data

    def _bm_toggle(self, value: bool) -> None:
        self._benchmark = value

        for fusion in self._bm_fusions:
            fusion["target"]._bm_toggle(value)

    @staticmethod
    def register(method: Callable, name=None) -> Callable:
        benchmark_name = method.__name__ if name is None else name

        def wrapper(self, *args, **kwargs):
            assert isinstance(self, Benchmarkable)

            self._bm(benchmark_name)
            result = method(self, *args, **kwargs)
            self._bm(benchmark_name)

            return result

        return wrapper
