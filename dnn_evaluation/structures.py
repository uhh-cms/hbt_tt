import numpy as np
import torch
import functools
import operator
from dataclasses import dataclass


@dataclass
class Process:
    """
    Class to store the datasets to process, together with its labels."""
    events: object
    description: str
    label: str

    def get_events(self, dataset):
        data = self.events[0][dataset]

        # Extrawurst for DY data
        # concatenate all dy events, which currently are stored as a dict:
        dy_indices = [k[1] for k in self.events[0][dataset].keys() if k[0] == "dy"]
        dicts = [(self.events[0][dataset][('dy', i)]) for i in dy_indices]

        # TODO: code is error-prone as new columns added to the NN output will not be adopted immediately
        events_dy = {
            "scores": torch.cat([d["scores"] for d in dicts]),
            "event_weight": torch.cat([d["event_weight"] for d in dicts]),
            "normalization_weights": torch.cat([d["normalization_weights"] for d in dicts]),
            "event_id": torch.cat([d["event_id"] for d in dicts]),
        }

        # store all categories in a dict:
        return {
            "events_tt_dl": data[("tt", 1200)],
            "events_tt_fh": data[("tt", 1300)],
            "events_tt_sl": data[("tt", 1100)],
            "events_hh": data[("hh", 21101)],
            "events_dy": events_dy
        }

@dataclass
class AnalysisResult:
    signal_hists: dict
    background_hists: dict
    binned_significances: dict
    total_significances: dict
    errors_significances: dict
    scaling_factor: float
