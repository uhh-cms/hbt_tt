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
from modules import logit, identity, asimov_significance, flats_binning, add_flow_bin
from structures import ProcessLoader, Process, HistFab
from matplotlib.patches import Rectangle

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
    process_loader.load_process(path_dnn+"/22_23_lr_linear_a_test_test.pt", label="22_23_lr_linear_a_test", description="(1,1,1)"),
    # process_loader.load_process(path_dnn+"/22_23_lr_lin_plus_cos_b_test.pt", label="22_23_lr_lin_plus_cos_b_test", description=r"Weighting: (1,1,1); LR = $10^{-3}$"),
      # sanity check: Marcel's old DNN
    #process_loader.load_process([path_old_dnn+"/tt_22pre_v14.parquet",
                                # path_old_dnn+"/hh_22pre_v14.parquet",
                                # path_old_dnn+"/dy_22pre_v14.parquet"
                                #],
                                 #label="input_file", description="original input file")
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
    
    ###
    # plot the matrix
    # TODO maybe implement a way to make changes from the baseline darker    
    fig, ax = plt.subplots(figsize=(5, 4))
    cell_color = "#B382C2"

    # Cells
    for i in range(3):
        for j in range(3):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor=cell_color, alpha=0.25, edgecolor="0.65", linewidth=0.8)
            )
            ax.text(
                j, i, f"{sig_frames.iloc[i, j]:.4f}", ha="center", va="center", fontsize=12
            )

    for j, col in zip([0,1,2], [r"$e\tau$", r"$\mu\tau$", r"$\tau\tau$"]):
        ax.text(
            (j + 0.5) / 3, 1.04, col, transform=ax.transAxes, ha="center", va="bottom", fontsize=11
        )

    ax.set_yticks(range(3))
    ax.set_yticklabels(sig_frames.index, fontsize=11)
    ax.set_xticks([]) # no x ticks as i did this manually in the loop above
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.tick_params(which="both", length=0)
    plt.tight_layout()
    plt.savefig(f"sig_matrix_images/sig_7{output.label}")
    plt.show()