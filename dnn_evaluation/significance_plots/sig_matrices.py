import itertools
from multiprocessing.util import debug
import torch
import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from termcolor import colored
from IPython import embed

import sys
sys.path.append("/afs/desy.de/user/h/hergesk/repos/hbt_tt/dnn_evaluation/modules")
from modules import logit, identity, asimov_significance, flats_binning, add_flow_bin
from hist_utils import ProcessLoader, Process, HistFab
from significance_utils import SigLoader

"""This script produces matrices depicting the significances of (etau, mutau, tautau) channel
and the three decay channels (dl, sl, fh).
"""
n_bins = 10
eps = 1e-6 # set eps=0 for normal scale
lower_border_flats = -1e2# set to 0 for lin scale
# upper_border = 12# set to 1 for lin scale
func = identity

label_color = "#7E3A72"# "#BE185D" ##'#4b2e83'
"#008C95"

colors = [
"#F28EBC",  # light pink
"#D45087",  # rose
"#9B5DE5",  # lavender violet
"#5A189A",  # dark violet
]


path_dnn = "/data/dust/user/hergesk/HH_DNN/evaluation"
path_old_dnn = "/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24"
process_loader = ProcessLoader()

data_dnn_outputs = [
    # my DNNs oversampling dl
    process_loader.load_process(path_dnn+"/tt_1_1_100_test.pt", label="baseline", description="(1,1,100)"),
    process_loader.load_process(path_dnn+"/tt_1p5_1_100_test.pt", label="1p5_1_100", description="(1.5,1,100)"),
    process_loader.load_process(path_dnn+"/tt_2_1_100_test.pt", label="2_1_100", description="(2,1,100)"),
    process_loader.load_process(path_dnn+"/tt_1_1p5_100_test.pt", label="1_1p5_100", description="(1,1.5,100)"),
    process_loader.load_process(path_dnn+"/tt_1_2_100_test.pt", label="1_2_100", description="(1,2,100)"),
    process_loader.load_process(path_dnn+"/tt_1p5_1p5_100_test.pt", label="1p5_1p5_100", description="(1.5,1.5,100)"),
    process_loader.load_process(path_dnn+"/tt_2_2_100_test.pt", label="2_2_100", description="(2,2,100)")
]

print(colored ("data loaded, starting analysis now.", "yellow"))
for output in data_dnn_outputs:
    print(f"Processing label {output.label}")
    hists = [HistFab("all_tt_hist", ["tt_dl", "tt_sl", "tt_fh"], "red", "tt: all events", flavor=output.flavor),
            HistFab("sl_hist", ["tt_sl"], "#009E73", "tt: sl events", flavor=output.flavor),
            HistFab("dl_hist", ["tt_dl"], "#0072B2", "tt: dl events", flavor=output.flavor),# or
            HistFab("fh_hist", ["tt_fh"], 'tab:brown', "tt: fh events", flavor=output.flavor),
            HistFab("dy_hist", ["dy"], "tab:orange", "dy: all events", flavor=output.flavor),# '#3B5B92'
            HistFab("hh_hist", ["hh"], "black", "hh: all events", flavor=output.flavor)
    ]
    # get bin edges for flat s binning
    if output.flavor == "torch_tensor":
        bin_edges = flats_binning(output.events["hh"]["scores"][:, 0], bin_num = n_bins, hist_edge_l=lower_border_flats)[2]
    if output.flavor == "ak_array":
        bin_edges = flats_binning(torch.from_numpy(ak.to_numpy(logit(output.events["hh"].run3_dnn_moe_hh))), bin_num = n_bins, hist_edge_l=lower_border_flats)[2]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    lower_border = bin_edges[0]
    upper_border = bin_edges[-1]
    histograms = {}
    for h in hists:
        histogram = h.create_hist_flats(bin_edges)
        h.fill_hist(
            histogram,
            func,
            output)
        histograms[h.name] = histogram
    tot_sig_all, tot_error_sig_all = asimov_significance(histograms["hh_hist"], histograms["dy_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")
    tot_sig_dl, tot_error_sig_dl = asimov_significance(histograms["hh_hist"], histograms["dl_hist"], error_type="poisson_weighted")
    tot_sig_sl, tot_error_sig_sl = asimov_significance(histograms["hh_hist"], histograms["sl_hist"], error_type="poisson_weighted")
    tot_sig_fh, tot_error_sig_fh = asimov_significance(histograms["hh_hist"], histograms["fh_hist"], error_type="poisson_weighted")
    all_significances = [tot_sig_all, tot_sig_dl, tot_sig_sl, tot_sig_fh]
    # all_errors = [tot_error_sig_all, tot_error_sig_dl, tot_error_sig_sl, tot_error_sig_fh]
    all_sig_tot = [np.sqrt(np.sum(np.square(s))) for s in all_significances]    
    split_events = [
        output.split_into_categories("etau"),
        output.split_into_categories("mutau"),
        output.split_into_categories("tautau")
    ]
    sig_cats = {"etau": {}, 
               "mutau": {},
               "tautau": {}
               }
    for d, category in zip(split_events, ["etau", "mutau", "tautau"]):
        # split the tt bg data in three processes
        # events_dict = d.get_events(dataset)
        
        hists = [HistFab("all_tt_hist", ["tt_dl", "tt_sl", "tt_fh"], "red", "tt: all events", flavor=d.flavor),
                    HistFab("sl_hist", ["tt_sl"], "#009E73", "tt: sl events", flavor=d.flavor),
                    HistFab("dl_hist", ["tt_dl"], "#0072B2", "tt: dl events", flavor=d.flavor),# or
                    HistFab("fh_hist", ["tt_fh"], 'tab:brown', "tt: fh events", flavor=d.flavor),
                    HistFab("dy_hist", ["dy"], "tab:orange", "dy: all events", flavor=d.flavor),# '#3B5B92'
                    HistFab("hh_hist", ["hh"], "black", "hh: all events", flavor=d.flavor)
        ]
        # get bin edges for flat s binning
        if d.flavor == "torch_tensor":
            bin_edges = flats_binning(d.events["hh"]["scores"][:, 0], bin_num = n_bins, hist_edge_l=lower_border_flats)[2]
        if d.flavor == "ak_array":
            bin_edges = flats_binning(torch.from_numpy(ak.to_numpy(logit(d.events["hh"].run3_dnn_moe_hh))), bin_num = n_bins, hist_edge_l=lower_border_flats)[2]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        lower_border = bin_edges[0]
        upper_border = bin_edges[-1]
        histograms = {}
        for h in hists:
            histogram = h.create_hist_flats(bin_edges)
            h.fill_hist(
                histogram,
                func,
                d)
            histograms[h.name] = histogram
        sig_all = np.sqrt(np.sum(np.square(asimov_significance(histograms["hh_hist"], histograms["dy_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")[0])))
        sig_dl = np.sqrt(np.sum(np.square(asimov_significance(histograms["hh_hist"], histograms["dl_hist"], error_type="poisson_weighted")[0])))
        sig_sl = np.sqrt(np.sum(np.square(asimov_significance(histograms["hh_hist"], histograms["sl_hist"], error_type="poisson_weighted")[0])))
        sig_fh = np.sqrt(np.sum(np.square(asimov_significance(histograms["hh_hist"], histograms["fh_hist"], error_type="poisson_weighted")[0])))
        
        sig_cats[category] = {"tt dl": round(sig_dl, 4),
                              "tt_sl": round(sig_sl, 4),
                              "tt_fh": round(sig_fh, 4)
                              }
    sig_frames = pd.DataFrame(sig_cats)
    if output.label == "baseline":
        sig_loader = SigLoader(sig_frames)
    sig_loader.make_matrix(sig_frames, save_path=f"images_sig_matrix/sig_{output.label}")