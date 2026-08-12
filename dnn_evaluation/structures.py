import numpy as np
import torch
import functools
import operator
import hist
from hist import Hist
from dataclasses import dataclass
from modules import flats_binning


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
        dy_indices = [k[1] for k in data.keys() if k[0] == "dy"]
        dicts = [(data[('dy', i)]) for i in dy_indices]

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

    def get_flats_binedges(self, dataset, n_bins=10):
        lower_border_flats = -1e2
        data = self.events[0][dataset]
        bin_edges = flats_binning(data[("hh", 21101)]["scores"][:, 0], bin_num = n_bins, hist_edge_l=lower_border_flats)[2]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # check if two bin edges are the same
        for i in range(len(bin_edges)):
            for j in range(len(bin_edges)):
                if (i != j) & (bin_edges[i] == bin_edges[j]):
                    print("\033[93mError: Two bin edges are the same! Check bin edges and delete one of the doubles!\033[0m")
        # important: map hist edges to bin edges from flat-s binning
        lower_border = bin_edges[0]
        upper_border = bin_edges[-1]
        return lower_border, upper_border, bin_edges, bin_centers

@dataclass
class HistFab:
    """
    Class to store histogram configurations and produce hists."""
    name: str
    event_keys: list[str]
    color: str
    label: str

    def get_hist_config(self):
        return {
            "name": self.name,
            "color": self.color,
            "label": self.label
        }

    def create_hist(self, n_bins, lower_border, upper_border):
        """version without flat-s binning."""
        return Hist(
            hist.axis.Regular(
                n_bins, lower_border, upper_border,
                name=self.name,
                label=self.label,
                underflow=True,
                overflow=True
            ),
            storage=hist.storage.Weight()
        )
    def create_hist_flats(self, bin_edges):
        """version with flat-s binning."""
        return Hist(
            hist.axis.Variable(
                bin_edges,
                name=self.name,
                label=self.label,
                flow=True),
            storage=hist.storage.Weight()
        )
    def fill_hist(self, h, func, values, weights):
        h.fill(
            func(values),
            weight =weights
        )
        return hist
    def reset_hist(self, *hists):
        for h in hists:
            h.reset()
