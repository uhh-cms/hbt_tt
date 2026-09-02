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

"""This script plots the difference in significance per bin from the baseline,
one plot for all significances, as a comparison."""

n_bins = 35
eps = 1e-6 # set eps=0 for normal scale
# plotting in logit space:
lower_border = 0
upper_border = 11
func = identity

path_dnn = "/data/dust/user/hergesk/HH_DNN/evaluation"
path_old_dnn = "/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24"
process_loader = ProcessLoader()

data_dnn_outputs = [
    # my DNNs oversampling dl
    process_loader.load_process(path_dnn+"/tt_1_1_100_test.pt", label="baseline", description="baseline"),
    process_loader.load_process(path_dnn+"/tt_1p5_1_100_test.pt", label="1p5_1_100", description="(1.5,1,100)"),
    process_loader.load_process(path_dnn+"/tt_2_1_100_test.pt", label="2_1_100", description="(2,1,100)"),
    process_loader.load_process(path_dnn+"/tt_1_1p5_100_test.pt", label="1_1p5_100", description="(1,1.5,100)"),
    process_loader.load_process(path_dnn+"/tt_1_2_100_test.pt", label="1_2_100", description="(1,2,100)"),
    process_loader.load_process(path_dnn+"/tt_1p5_1p5_100_test.pt", label="1p5_1p5_100", description="(1.5,1.5,100)"),
    process_loader.load_process(path_dnn+"/tt_2_2_100_test.pt", label="2_2_100", description="(2,2,100)")
]

all_sigs_per_bin = {}
all_sigs_tt_per_bin = {}

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
    histograms = {}
    for h in hists:
        histogram = h.create_hist(n_bins, lower_border, upper_border)
        h.fill_hist(
            histogram,
            func,
            output.add_btagcut())
        histograms[h.name] = histogram
    tot_sig_all_binned, tot_error_sig_all_binned = asimov_significance(histograms["hh_hist"], histograms["dy_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")
    tot_sig_tt_binned, tot_error_sig_tt_binned = asimov_significance(histograms["hh_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")

    tot_sig_all = np.sqrt(np.sum(np.square(tot_sig_all_binned)))
    tot_sig_tt = np.sqrt(np.sum(np.square(tot_sig_tt_binned)))
    all_sigs_per_bin[output.description] = {"per_bin":      tot_sig_all_binned,
                                            "err_per_bin":  tot_error_sig_all_binned,
                                            "total": tot_sig_all}
    all_sigs_tt_per_bin[output.description] = {"per_bin":      tot_sig_tt_binned,
                                        "err_per_bin":  tot_error_sig_tt_binned,
                                        "total": tot_sig_tt}
    
x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers
fig, ax = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)

### -----
# plot all (tt+dy) sigs in one plot
for key in all_sigs_per_bin.keys():
    ax.errorbar(x_lin_bincenters, all_sigs_per_bin[key]["per_bin"] - all_sigs_per_bin["baseline"]["per_bin"], 
                #yerr=all_sigs_per_bin[key]["err_per_bin"], 
                label=key+fr"; total: {round(all_sigs_per_bin[key]['total'], 2)}", 
                alpha=1.0, elinewidth=0.5, capsize=2)# , errorevery=2

ax.set_xlabel("DNN output node score")
ax.set_ylabel(r"$\Delta Z_A$ = $Z_{A, DNN} - Z_{A, baseline}$")
# ax.set_xscale("log")

plt.legend()
plt.title("Relative difference in Asimov significance (tt + dy)for all tested DNN's")
plt.savefig(f"images_all_sigs/all_delta_sig_ttdy", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

### -----
# plot all tt sigs in one plot
fig, ax = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
for key in all_sigs_tt_per_bin.keys():
    ax.errorbar(x_lin_bincenters, all_sigs_tt_per_bin[key]["per_bin"] - all_sigs_tt_per_bin["baseline"]["per_bin"], 
                #yerr=all_sigs_tt_per_bin[key]["err_per_bin"], 
                label=key+fr"; total: {round(all_sigs_tt_per_bin[key]['total'], 2)}", 
                alpha=1.0, elinewidth=0.5, capsize=2)# , errorevery=2

ax.set_xlabel("DNN output node score")
ax.set_ylabel(r"$\Delta Z_A$ = $Z_{A, DNN} - Z_{A, baseline}$")
# ax.set_xscale("log")

plt.legend()
plt.title("Relative difference in Asimov significance (only tt) for all tested DNN's")
plt.savefig(f"images_all_sigs/all_delta_sig_tt", dpi=300, bbox_inches='tight')
plt.show()