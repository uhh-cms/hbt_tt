import numpy as np
import torch
import functools
import operator
import hist
from hist import Hist
from dataclasses import dataclass
from modules import flats_binning

from modules import logit

class ProcessAgregator:
    """
    collect signal and background events from two possible input types: pytorch tensor and columflow ak array
    """
    def __init__(self) -> None:
        self.processes = {}

    def register_process(self, array, flavor, label, description, **kwargs):
        if flavor == "cf":
            self._register_columnflow_array(array, **kwargs)
        elif flavor == "torch":
            self._register_pt_array(array=array, label=label, description=description, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")


    def get_process(self, label):
        pass

    def _register_pt_array(self, array, label, description):
        data = array
        dy_indices = [] # collect all dy indices
        # dy_dict = {} # new dict to store all dy data together
        for dataset in ["training", "validation", "test"]:
            dy_indices = [k[1] for k in data[0][dataset].keys() if k[0] == "dy"]
            dicts = [(data[0][dataset][('dy', i)]) for i in dy_indices]

        # TODO: code is error-prone as new columns added to the NN output will not be adopted immediately
            events_dy = {
                "scores": torch.cat([d["scores"] for d in dicts]),
                "event_weight": torch.cat([d["event_weight"] for d in dicts]),
                "normalization_weights": torch.cat([d["normalization_weights"] for d in dicts]),
                "event_id": torch.cat([d["event_id"] for d in dicts]),
            }

        # store all categories in a dict:
        data = {
            "tt_dl": data[0][dataset][("tt", 1200)],
            "tt_fh": data[0][dataset][("tt", 1300)],
            "tt_sl": data[0][dataset][("tt", 1100)],
            "hh": data[0][dataset][("hh", 21101)],
            "dy": events_dy
        }

        for name, events in data.items():
            _process_type = name.split("_")[0]
            process = Process(
                events = data[name],
                subprocess = name,
                label = label,
                process_type = _process_type
            )
            self.processes[name] = process

    def _register_columnflow_array(self, data, **kwargs):
        filter_default = data.run3_dnn_moe_hh > 0
        data = data[filter_default]

        # default plotting is in logit space; otherwise change func to identity
        # convert_to_logit = lambda x: func(x.run3_dnn_moe_hh)

        unique_process_id = list(sorted(np.unique(data.process_id)))
        events = {}
        if 1100 in unique_process_id:
            label = "tt"
            events["tt_dl"] = data[data.process_id == 1200]
            events["tt_fh"] = data[data.process_id == 1300]
            events["tt_sl"] = data[data.process_id == 1100]

        elif 21101 in unique_process_id:
            events["hh"] = data[data.process_id == 21101]
            label = "hh"
        else:
            label = "dy"
            events["dy"] = data
        from IPython import embed; embed(header="MAYBE LAST TODO: fix how the different processes are concatenated! Line 88 | File: structures.py")
        for name , _events in events.items():
            process = Process(
                events = _events,
                subprocess = name,
                label = label,
                process_type = name.split("_")[0]
            )
            self.processes[name] = process

    def get_process_name_from_id(id):
        # TODO use global id to name matching
        labels = {
            1300 : "events_tt_fh",
            1200 : "events_tt_dl",
            1100 : "events_tt_sl",
            21101 : "events_hh",
        }
        return labels.get(id, "events_dy")





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
class Process:
    """
    Class to store the datasets to process, together with its labels."""
    events: object
    subprocess: str
    label: str
    process_type: str




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
            "label": self.label,
            "type": self.type
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
