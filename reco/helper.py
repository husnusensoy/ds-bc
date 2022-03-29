import numpy as np
from typing import Dict, Any, List
from itertools import product


class FrequencyDist:
    counter: Dict[Any, int]

    def __init__(self):
        self.counter = {}

    def __getitem__(self, item) -> int:
        if item in self.counter:
            return self.counter[item]
        else:
            return 0

    def __setitem__(self, key, value: int):
        self.counter[key] = value

    def __repr__(self):
        return str(self.counter)

    @property
    def total(self) -> int:
        return sum(self.counter.values())


class CondFrequencyDist:
    freq_dist: Dict[Any, FrequencyDist]

    def __init__(self):
        self.freq_dist = {}

    def __getitem__(self, item) -> FrequencyDist:
        if item not in self.freq_dist:
            self.freq_dist[item] = FrequencyDist()

        return self.freq_dist[item]

    def __repr__(self):
        return str(self.freq_dist)

    @property
    def total(self) -> int:
        return sum(self.counter.values())


class ProbabilityDist:
    freq: FrequencyDist

    def __init__(self, freq: FrequencyDist, events: List[Any] = None, add: int = 1):
        self.freq = freq

        if events is not None:
            for e in events:
                self.freq[e] += add

    def __getitem__(self, item) -> float:
        return self.freq[item] / self.freq.total

    def __repr__(self):
        return str(dict((k, self[k]) for k in self.freq.counter))


class CondProbabilityDist:
    proba_dist: Dict[Any, ProbabilityDist]

    def __init__(self, cond_freq: CondFrequencyDist, events: List[Any] = None, add: int = 1):
        self.proba_dist = dict((c, ProbabilityDist(fdist, events, add)) for c, fdist in cond_freq.freq_dist.items())

    def __getitem__(self, item) -> ProbabilityDist:
        return self.proba_dist[item]

    def __repr__(self):
        return str(self.proba_dist)