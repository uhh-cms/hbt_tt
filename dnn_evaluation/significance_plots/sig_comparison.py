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

"""This script plots the different significances,
in one plot using a flat-s binning."""

n_bins = 10
eps = 1e-6 # set eps=0 for normal scale
# plotting in logit space:
lower_border = 0
upper_border = 13
func = identity
lower_border_flats = -50

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
# for output in data_dnn_outputs:
#     print(f"Processing label {output.label}")
#     hists = [HistFab("all_tt_hist", ["tt_dl", "tt_sl", "tt_fh"], "red", "tt: all events", flavor=output.flavor),
#             HistFab("sl_hist", ["tt_sl"], "#009E73", "tt: sl events", flavor=output.flavor),
#             HistFab("dl_hist", ["tt_dl"], "#0072B2", "tt: dl events", flavor=output.flavor),# or
#             HistFab("fh_hist", ["tt_fh"], 'tab:brown', "tt: fh events", flavor=output.flavor),
#             HistFab("dy_hist", ["dy"], "tab:orange", "dy: all events", flavor=output.flavor),# '#3B5B92'
#             HistFab("hh_hist", ["hh"], "black", "hh: all events", flavor=output.flavor)
#     ]
#     # flat-s binning
#     from IPython import embed; embed(header="DEBUG ALL SIG SCRIPT WITH FLATS Line 60 | File: sig_comparison2.py")
#     bin_edges = flats_binning(output.events["hh"]["scores"][:, 0], bin_num = n_bins, hist_edge_l=lower_border_flats)[2]
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#     lower_border = bin_edges[0]
#     upper_border = bin_edges[-1]

#     histograms = {}
#     for h in hists:
#         histogram = h.create_hist_flats(bin_edges)
#         h.fill_hist(
#             histogram,
#             func,
#             output.add_btagcut())
#         histograms[h.name] = histogram
#     tot_sig_all_binned, tot_error_sig_all_binned = asimov_significance(histograms["hh_hist"], histograms["dy_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")

#     tot_sig_all = np.sqrt(np.sum(np.square(tot_sig_all_binned)))
#     all_sigs_per_bin[output.description] = {"per_bin":      tot_sig_all_binned,
#                                             "err_per_bin":  tot_error_sig_all_binned,
#                                             "total": tot_sig_all}

# x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
# x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers
# fig, ax = plt.subplots(figsize=(9, 5))
# fig.subplots_adjust(right=0.85)

# color_lib = {
#     "baseline": "black",
#     '(1.5,1,100)': "red",
#     "(2,1,100)": "blue",
#     "(1,1.5,100)": "green",
#     "(1,2,100)": "orange",
#     "(1.5,1.5,100)": "purple",
#     "(2,2,100)": "brown"
# }

# # plot all sigs in one plot
# for key in all_sigs_per_bin.keys():
#     if key == "baseline":
#         alpha = 1.0
#     else:
#         alpha = 0.6
#     ax.errorbar(x_lin_bincenters, all_sigs_per_bin[key]["per_bin"] - all_sigs_per_bin["baseline"]["per_bin"],
#                 #yerr=all_sigs_per_bin[key]["err_per_bin"],
#                 label=key+fr"; total: {round(all_sigs_per_bin[key]['total'], 2)}",
#                 color=color_lib[key],
#                 alpha=alpha,
#                 elinewidth=0.5, capsize=2)# , errorevery=2

# ax.set_xlabel("DNN output node score")
# ax.set_ylabel(r"$\Delta Z_A$ = $Z_{A, DNN} - Z_{A, baseline}$")
# # ax.set_xscale("log")

# plt.legend()
# plt.title("Relative difference in Asimov significance (tt + dy)for all tested DNN's, res1b + res2b")
# plt.savefig(f"images_all_sigs/all_delta_sig_ttdy", dpi=300, bbox_inches='tight')
# plt.show()
# plt.clf()

### -----
for output in data_dnn_outputs:
    print(f"Processing label {output.label}")
    hists = [HistFab("all_tt_hist", ["tt_dl", "tt_sl", "tt_fh"], "red", "tt: all events", flavor=output.flavor),
            HistFab("sl_hist", ["tt_sl"], "#009E73", "tt: sl events", flavor=output.flavor),
            HistFab("dl_hist", ["tt_dl"], "#0072B2", "tt: dl events", flavor=output.flavor),# or
            HistFab("fh_hist", ["tt_fh"], 'tab:brown', "tt: fh events", flavor=output.flavor),
            HistFab("dy_hist", ["dy"], "tab:orange", "dy: all events", flavor=output.flavor),# '#3B5B92'
            HistFab("hh_hist", ["hh"], "black", "hh: all events", flavor=output.flavor)
    ]

    # flat-s binning
    bin_edges = flats_binning(output.events["hh"]["scores"][:, 0], bin_num = n_bins, hist_edge_l=lower_border_flats)[2]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    lower_border = bin_edges[0]
    upper_border = bin_edges[-1]

    histograms = {}
    for h in hists:
        histogram = h.create_hist_flats(bin_edges)
        h.fill_hist(
            histogram,
            func,
            output.add_btagcut())
        histograms[h.name] = histogram
    tot_sig_all_binned, tot_error_sig_all_binned = asimov_significance(histograms["hh_hist"], histograms["dy_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")

    # save min and max y value for plotting
    min_y = np.min(tot_sig_all_binned)# - tot_error_sig_all_binned[0])
    max_y = np.max(tot_sig_all_binned)# + tot_error_sig_all_binned[1])

    tot_sig_all = np.sqrt(np.sum(np.square(tot_sig_all_binned)))
    all_sigs_per_bin[output.description] = {"per_bin":      tot_sig_all_binned,
                                            "err_per_bin":  tot_error_sig_all_binned,
                                            "total": tot_sig_all,
                                            "ylim": (min_y, max_y)}


x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers
fig, ax = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)

# plot all (tt+dy) sigs in one plot
min_y = 0
max_y = 0
color_lib = {
    "baseline": "black",
    "(1.5,1,100)": "red",
    "(2,1,100)": "blue",
    "(1,1.5,100)": "green",
    "(1,2,100)": "orange",
    "(1.5,1.5,100)": "purple",
    "(2,2,100)": "brown"
}
for key in all_sigs_per_bin.keys():
    if key == "baseline":
        alpha = 1.0
    else:
        alpha = 0.6
    # plt.plot(x_lin_bincenters, all_sigs_per_bin[key]["per_bin"], label=key+fr"; total: {round(all_sigs_per_bin[key]['total'], 2)}", color=color_lib[key], alpha=alpha)
    ax.errorbar(x_lin_bincenters, all_sigs_per_bin[key]["per_bin"],
                yerr=all_sigs_per_bin[key]["err_per_bin"],
                label=key+fr"; total: {round(all_sigs_per_bin[key]['total'], 2)}",
                elinewidth=0.5, capsize=2, color=color_lib[key], alpha=alpha)# , errorevery=2
    min_y = min(min_y, all_sigs_per_bin[key]['ylim'][0])
    max_y = max(max_y, all_sigs_per_bin[key]['ylim'][1])

plt.ylim(max(min_y, 1e-3), max_y)
ax.set_xlabel("DNN output node score", labelpad=16)
ax.text(
    1.0, -0.08,
    "Bin number",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize = 12
)
ax.set_ylabel(r"$Z_A$")
ax.yaxis.set_label_coords(-0.08, 0.94)
ax.set_xticks(x_lin_binedges)  # Set label locations.
ax.set_xticklabels(np.arange(0,11,1))  # Set text labels.
# ax.set_yscale("function", functions=(np.sqrt, lambda x: x**2))
# ax.set_yscale("logit")
# ax.set_xscale("log")

plt.legend()
plt.title("Asimov significance (tt + dy) for all tested DNN's, res1b + res2b, flat-s binning")
plt.savefig(f"all_sigs/all_sig_ttdy", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
