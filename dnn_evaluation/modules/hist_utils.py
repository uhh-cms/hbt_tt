import numpy as np
import torch
import functools
import operator
import hist
from hist import Hist
from dataclasses import dataclass
from pathlib import Path
import awkward as ak
from termcolor import colored

import sys
sys.path.append("/afs/desy.de/user/h/hergesk/repos/hbt_tt/dnn_evaluation/modules")
from modules import logit, flats_binning

from IPython import embed

class ProcessLoader:
    """
    load signal and background events from two possible input types: pytorch tensor and columflow ak array
    and bring it in the shape we need:
    - hh: signal
    - dy: background
    - tt: background split in W decay mode
        -> tt dl
        -> tt sl
        -> tt fh

    Input: Data path and label (torch tensor and ak array origin both allowed)
    Output: sorted data
    """

    def __init__(self):
        self.processes = {}

    @staticmethod
    def get_flavor(path):
        flavor = path.split(".")[-1]
        return flavor

    def load_process(self, path, label, description):
        # if input is only one str, turn it into array
        if isinstance(path, (str, Path)):
            path = [path]
        flavor = self.get_flavor(path[0])
        if flavor == "parquet":
            events = self._register_columnflow_array(path, label, description)
            return events
        elif flavor == "pt":
            events = self._register_pt_array(path[0], label, description)
            return events
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _register_pt_array(self, path, label, description):
        data = torch.load(path, map_location=torch.device("cpu"))
        dy_indices = []  # collect all dy indices
        # dy_dict = {} # new dict to store all dy data together
        dy_indices = [k[1] for k in data[0].keys() if k[0] == "dy"]
        dicts = [(data[0][("dy", i)]) for i in dy_indices]
        # TODO: code is error-prone as new columns added to the NN output will not be adopted immediately
        events_dy = {
            "scores": torch.cat([d["scores"] for d in dicts]),
            # skip event_weight as they're both in the product_of_weights column now
            # "event_weight": torch.cat([d["event_weight"] for d in dicts]),
            "normalization_weights": torch.cat(
                            [d["normalization_weights"] for d in dicts]
            ),
            "product_of_weights": torch.cat(
                [d["product_of_weights"] for d in dicts]
            ),
            "event_id": torch.cat([d["event_id"] for d in dicts]),
            "pair_type": torch.cat([d["pair_type"] for d in dicts]),
            "bjet_mask": torch.cat([d["bjet_mask"] for d in dicts]),
            "di_bjet": torch.cat([d["di_bjet"] for d in dicts])
        }

        # store all categories in a dict:
        data = {
            "tt_dl": data[0][("tt", 1200)],
            "tt_fh": data[0][("tt", 1300)],
            "tt_sl": data[0][("tt", 1100)],
            "hh": data[0][("hh", 21101)],
            "dy": events_dy,
        }
        return Process(
            events=data,
            label=label,
            description=description,
            flavor="torch_tensor"
        )

    def _register_columnflow_array(self, path, label, description):
        events = {}
        for p in path:
            data = ak.from_parquet(p)
            filter_default = data.run3_dnn_moe_hh > 0
            data = data[filter_default]
            # default plotting is in logit space; otherwise change func to identity
            # convert_to_logit = lambda x: func(x.run3_dnn_moe_hh)

            unique_process_id = list(sorted(np.unique(data.process_id)))
            if 1100 in unique_process_id:
                tag = "tt"
                events["tt_dl"] = data[data.process_id == 1200]
                events["tt_fh"] = data[data.process_id == 1300]
                events["tt_sl"] = data[data.process_id == 1100]

            elif 21101 in unique_process_id:
                events["hh"] = data[data.process_id == 21101]
                tag = "hh"
            else:
                tag = "dy"
                events["dy"] = data
        return Process(
            events=events,
            label=label,
            description=description,
            flavor="ak_array"
        )

    def get_flats_binedges(self, dataset, n_bins=10):
        "work in progress, this fct is not used yet."
        lower_border_flats = -1e2
        data = self.events[0][dataset]
        bin_edges = flats_binning(
            data[("hh", 21101)]["scores"][:, 0],
            bin_num=n_bins,
            hist_edge_l=lower_border_flats,
        )[2]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # check if two bin edges are the same
        for i in range(len(bin_edges)):
            for j in range(len(bin_edges)):
                if (i != j) & (bin_edges[i] == bin_edges[j]):
                    print(
                        "\033[93mError: Two bin edges are the same! Check bin edges and delete one of the doubles!\033[0m"
                    )
        # important: map hist edges to bin edges from flat-s binning
        lower_border = bin_edges[0]
        upper_border = bin_edges[-1]
        return lower_border, upper_border, bin_edges, bin_centers
    

            


@dataclass
class Process:
    """
    Class to store the datasets to process, together with its labels."""

    events: object
    label: str
    description: str
    flavor: str
    
    def split_into_categories(self, cat):
        """splits data in the 3 categories:
        - etau
        - mutau
        - tautau
        and adds a btag cut (res1b + res2b together).
        """
        if cat == "mutau":
            filter_index = 0           
        elif cat == "etau":
            filter_index = 1
        elif cat == "tautau":
            filter_index = 2
        else:
            print(colored("Warning: category to split doesn't match the possible options. Please use 'etau', 'mutau' or 'tautau'.", "red"))
        ev = {
            key: event.copy()
            for key, event in self.events.items()
        }
        
        for key, event in ev.items():
            # keys are masks, event are "hh", "tt_dl" etc  
            
            # define masks (bmask2 already includes bmask1)
            bmask1 = event["bjet_mask"]
            bmask2 = event["di_bjet"]
            btag_mask = bmask1 | bmask2

            pairtype_mask = event["pair_type"] == filter_index
            mask =  btag_mask & pairtype_mask

            for field, value in event.items():
                if field not in ("bjet_mask", "di_bjet"):
                    event[field] = value[mask]

        return Process(
                    events=ev,
                    label=self.label,
                    description=self.description,
                    flavor=self.flavor
                )
    def add_btagcut(self):
        """adds a btag cut (res1b + res2b together).
        """
        ev = {
            key: event.copy()
            for key, event in self.events.items()
        }
        
        for key, event in ev.items():
            # keys are masks, event are "hh", "tt_dl" etc  
            
            # define masks (bmask2 already includes bmask1)
            bmask1 = event["bjet_mask"]
            bmask2 = event["di_bjet"]
            btag_mask = bmask1 | bmask2

            for field, value in event.items():
                if field not in ("bjet_mask", "di_bjet"):
                    event[field] = value[btag_mask]

        return Process(
                    events=ev,
                    label=self.label,
                    description=self.description,
                    flavor=self.flavor
                )


@dataclass
class HistFab:
    """
    Class to store histogram configurations and produce hists."""

    name: str
    event_keys: list[str]
    color: str
    label: str
    flavor: str

    def get_hist_config(self):
        return {
            "name": self.name,
            "color": self.color,
            "label": self.label,
            "type": self.type,
        }

    def create_hist(self, n_bins, lower_border, upper_border):
        """version without flat-s binning."""
        return Hist(
            hist.axis.Regular(
                n_bins,
                lower_border,
                upper_border,
                name=self.name,
                label=self.label,
                underflow=True,
                overflow=True,
            ),
            storage=hist.storage.Weight(),
        )

    def create_hist_flats(self, bin_edges):
        """version with flat-s binning."""
        return Hist(
            hist.axis.Variable(bin_edges, name=self.name, label=self.label, flow=True),
            storage=hist.storage.Weight(),
        )

    def fill_hist(self, h, func, events):
        from modules import logit, identity
        if self.flavor == "torch_tensor":
            for key in self.event_keys:
                h.fill(
                    func(events.events[key]["scores"].numpy()[:, 0]),
                    weight=events.events[key]["product_of_weights"].numpy() * events.events[key]["normalization_weights"].numpy()
                )
            if func == logit:
                print(colored("Warning: logit plots first need to be implemented in HistFab class", "red"))
        if self.flavor == "ak_array":
            for key in self.event_keys:
                h.fill(
                    logit(events.events[key].run3_dnn_moe_hh),
                    weight=events.events[key].event_weight
                )

        return hist

    def reset_hist(self, *hists):
        for h in hists:
            h.reset()
