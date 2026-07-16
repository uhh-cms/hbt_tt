import itertools
import torch
import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
from modules import logit, identity, asimov_significance, def_equbin

"""This script analyses the first trained DNN's which sample the tt background in different ways concerning their W decay mode,
meaning di-leptonic, semi-leptonic and full-hadronic W decay.
"""
n_bins = 20
eps = 1e-6 # set eps=0 for normal scale
lower_border = -14# set to 0 for lin scale
# upper_border = 12# set to 1 for lin scale
func = identity

# output from my own DNNs
events_reference = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/referenz_dl1.pt", map_location=torch.device('cpu'))
events_train_dl2 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl2.pt", map_location=torch.device('cpu'))
events_train_dl4 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl4.pt", map_location=torch.device('cpu'))
events_train_dl6 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl6.pt", map_location=torch.device('cpu'))

# oversampling sl too:
events_train_dl2sl2 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl2sl2.pt", map_location=torch.device('cpu'))
events_train_dl1sl2 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl1sl2.pt", map_location=torch.device('cpu'))

# events_train = events_train[events_train.run3_dnn_moe_hh > 0]
for events, label1, label2 in zip([events_reference, events_train_dl4, events_train_dl2, events_train_dl6, events_train_dl2sl2, events_train_dl1sl2],
                                  ["equal sampling of W decay modes: (1,1,1)",
                                    "dl W decay mode oversampled: (1,1,2)",
                                    "dl W decay mode oversampled: (1,1,4)",
                                    "dl W decay mode oversampled: (1,1,6)",
                                    "dl and sl W decay mode oversampled: (1,2,2)",
                                    "sl W decay mode oversampled: (1,2,1)"],
                                    [111, 112, 114, 116, 122, 121]):
# split the tt bg data in three processes
    events_tt_dl = events[0]["test"][('tt', 1200)]
    events_tt_fh = events[0]["test"][('tt', 1300)]
    events_tt_sl = events[0]["test"][('tt', 1100)]
    events_hh    = events[0]["test"][('hh', 21101)]
    # concatenate all dy events, which currently are stored as a dict:
    dy_indices = [k[1] for k in events[0]["test"].keys() if k[0] == "dy"]
    dicts = [(events[0]["test"][('dy', i)]) for i in dy_indices]
    # TODO: code is error-prone as new columns added to the NN output will not be adopted immediately
    events_dy = {
        "scores": torch.cat([d["scores"] for d in dicts]),
        "event_weight": torch.cat([d["event_weight"] for d in dicts]),
        "normalization_weights": torch.cat([d["normalization_weights"] for d in dicts]),
        "event_id": torch.cat([d["event_id"] for d in dicts]),
    }
    # get bin edges for flat s binning
    bin_edges = def_equbin(events_hh["scores"][:, 0], bin_num = n_bins-1, hist_edge_l=lower_border)[2]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # check if two bin edges are the same
    for i in range(len(bin_edges)):
        for j in range(len(bin_edges)):
            if (i != j) & (bin_edges[i] == bin_edges[j]):
                print("\033[93mError: Two bin edges are the same! Check bin edges and delete one of the doubles!\033[0m")
    # important: map hist edges to bin edges from flat-s binning
    lower_border = bin_edges[0]
    upper_border = bin_edges[-1]
    # initialize hists with flat s binning edges
    # hist.axis.variable(bin_edges, name="", label=r"")

    dl_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r"", flow=True))
    fh_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r"", flow=True))
    sl_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r"", flow=True))
    dy_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r"", flow=True))
    hh_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r"", flow=True))
    all_tt_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r"", flow=True))
    # fill
    dl_hist.fill(func(events_tt_dl["scores"].numpy()[:, 0]), weight =events_tt_dl["event_weight"].numpy()* events_tt_dl["normalization_weights"].numpy())
    fh_hist.fill(func(events_tt_fh["scores"].numpy()[:, 0]), weight =events_tt_fh["event_weight"].numpy()* events_tt_fh["normalization_weights"].numpy())
    sl_hist.fill(func(events_tt_sl["scores"].numpy()[:, 0]), weight =events_tt_sl["event_weight"].numpy()* events_tt_sl["normalization_weights"].numpy())
    dy_hist.fill(func(events_dy["scores"].numpy()[:, 0]), weight =events_dy["event_weight"].numpy()* events_dy["normalization_weights"].numpy())
    hh_hist.fill(func(events_hh["scores"].numpy()[:, 0]), weight =events_hh["event_weight"].numpy()* events_hh["normalization_weights"].numpy())
    all_tt_hist.fill(func(events_tt_dl["scores"].numpy()[:, 0]), weight =events_tt_dl["event_weight"].numpy()* events_tt_dl["normalization_weights"].numpy())
    all_tt_hist.fill(func(events_tt_fh["scores"].numpy()[:, 0]), weight =events_tt_fh["event_weight"].numpy()* events_tt_fh["normalization_weights"].numpy())
    all_tt_hist.fill(func(events_tt_sl["scores"].numpy()[:, 0]), weight =events_tt_sl["event_weight"].numpy()* events_tt_sl["normalization_weights"].numpy())

    sig_all, error_sig_all = asimov_significance(hh_hist, dy_hist, fh_hist, dl_hist, sl_hist)
    sig_dl, error_sig_dl = asimov_significance(hh_hist, dl_hist)
    sig_sl, error_sig_sl = asimov_significance(hh_hist, sl_hist)
    sig_fh, error_sig_fh = asimov_significance(hh_hist, fh_hist)
    all_significances = [sig_all, sig_dl, sig_sl, sig_fh]
    all_errors = [error_sig_all, error_sig_dl, error_sig_sl, error_sig_fh]
    all_sig_tot = [np.sqrt(np.sum(np.square(s))) for s in all_significances]
    scaling_factor = ((hh_hist.values().sum())/ (all_tt_hist.values().sum() + dy_hist.values().sum()))**(-1)
    # scaling_factor = ((hh_hist.values().sum())/ (all_tt_hist.values().sum() ))**(-1)
    # plot
    x = bin_edges  # bin edges
    x_bin_centers = (x[:-1] + x[1:]) / 2  # bin centers
    x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
    x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.stairs(hh_hist.values()*scaling_factor, edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor="black", label=fr"signal x {round(scaling_factor)}")
    ax1.stairs(all_tt_hist.values(), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='red', label=r"tt: all events")
    ax1.stairs(sl_hist.values(), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='green', label=r"tt: sl decay")
    ax1.stairs(dl_hist.values(), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='blue', label=r"tt: dl decay")
    ax1.stairs(fh_hist.values(), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:brown', label=r"tt: fh decay")
    ax1.stairs(dy_hist.values(), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:orange', label=r"dy")
    ax1.tick_params(axis='y', labelcolor='black')

    yaxis_sig = ax1.twinx()
    label_color = '#4b2e83'
    colors = [
    "#F28EBC",  # light pink
    "#D45087",  # rose
    "#9B5DE5",  # lavender violet
    "#5A189A",  # dark violet
    ]
    yaxis_sig.set_ylabel(r'asimov significance', color=label_color)
    for sig, error, sig_tot, label, color in zip(all_significances,
                                            all_errors,
                                            all_sig_tot,
                                            ["tt + dy sig", "tt dl sig", "tt sl sig", "tt fh sig"],
                                            colors):
        yaxis_sig.errorbar(x_lin_bincenters, sig, yerr=error, label=label+f"; total: {round(sig_tot, 2)}", color=color, alpha=1.0)
    yaxis_sig.tick_params(axis='y', labelcolor=label_color)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = yaxis_sig.get_legend_handles_labels()
    ax1.set_yscale("log")
    yaxis_sig.set_yscale("log")
    ax1.set_ylabel('Number of events', color="black")

    # lower x axis with bin edges
    ax1.set_xticks(x_lin_binedges)  # Set label locations.
    ax1.set_xticklabels(x.round(2), rotation=45)  # Set text labels.
    ax1.set_xlabel('HH output node')

    # upper x axis with bin numbers
    ax1_upper = ax1.twiny()
    ax1_upper.set_xlim(ax1.get_xlim())
    ax1_upper.set_xticks(x_lin_bincenters)
    ax1_upper.set_xticklabels(range(0,len(x_lin_binedges)-1), rotation=0)  # Set text labels.
    ax1_upper.set_xlabel('bin number')

    yaxis_sig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.45, 1))
    # plt.legend(fontsize="small")
    # ax1.set_ylim(bottom=1e-1)
    # fig.tight_layout()
    plt.title(fr"HH output node for signal and tt background; {label1}; flat-s binning; total significance: {round(all_sig_tot[0], 5)}", wrap=True, pad=13)
    plt.savefig(f"images/first_training_{label2}_ttdy", dpi=300, bbox_inches='tight')
    plt.show()


    sl_hist.reset()
    dl_hist.reset()
    fh_hist.reset()
    dy_hist.reset()
    all_tt_hist.reset()
    hh_hist.reset()

################################################################################################
# sanity-check: "old data"
# func = logit
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/tt_22pre_v14.parquet")
# events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/hh_22pre_v14.parquet")
# events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/dy_22pre_v14.parquet")
# events_tt_dl = events_tt[events_tt.process_id == 1200]
# events_tt_fh = events_tt[events_tt.process_id == 1300]
# events_tt_sl = events_tt[events_tt.process_id == 1100]

# # get bin edges for flat s binning
# hh_array_torch = torch.from_numpy(ak.to_numpy(events_hh.run3_dnn_moe_hh))
# bin_edges = def_equbin(hh_array_torch, bin_num = n_bins-1, hist_edge_l=lower_border)[2]
# lower_border = bin_edges[0]
# upper_border = bin_edges[-1]
# # check if two bin edges are the same
# for i in range(len(bin_edges)):
#     for j in range(len(bin_edges)):
#         if (i != j) & (bin_edges[i] == bin_edges[j]):
#             print("\033[93mError: Two bin edges are the same! Check bin edges and delete one of the doubles!\033[0m")
# # initialize hists with flat s binning edges
# # hist.axis.variable(bin_edges, name="", label=r"")
# dl_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r""))
# fh_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r""))
# sl_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r""))
# dy_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r""))
# hh_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r""))
# all_tt_hist = Hist(hist.axis.Variable(bin_edges, name="", label=r""))
# # fill
# dl_hist.fill(func(events_tt_dl.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_tt_dl.event_weight)
# fh_hist.fill(func(events_tt_fh.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_tt_fh.event_weight)
# sl_hist.fill(func(events_tt_sl.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_tt_sl.event_weight)
# dy_hist.fill(func(events_dy.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_dy.event_weight)
# hh_hist.fill(func(events_hh.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_hh.event_weight)
# all_tt_hist.fill(func(events_tt_dl.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_tt_dl.event_weight)
# all_tt_hist.fill(func(events_tt_fh.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_tt_fh.event_weight)
# all_tt_hist.fill(func(events_tt_sl.run3_dnn_moe_hh, lower_border=lower_border, upper_border=upper_border), weight =events_tt_sl.event_weight)

# sig_all = asimov_significance(hh_hist, dy_hist, fh_hist, dl_hist, sl_hist)
# sig_no_dy = asimov_significance(hh_hist, fh_hist, dl_hist, sl_hist)
# total_significance = np.sqrt(np.sum(np.square(sig_all)))
# scaling_factor = ((hh_hist.values().sum())/ (all_tt_hist.values().sum() + dy_hist.values().sum()))**(-1)
# # plot
# x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
# x = (x[:-1] + x[1:]) / 2  # bin centers
# fig, ax1 = plt.subplots(figsize=(9, 5))
# fig.subplots_adjust(right=0.85)
# ax1.set_xlabel('HH output node')
# ax1.set_ylabel('Number of events', color="black")
# ax1.step(x, hh_hist.values()* scaling_factor, alpha=0.9, label=fr"signal x {round(scaling_factor)}", color="black")
# ax1.step(x, all_tt_hist.values(), alpha=0.9, label=r"tt: all events", color='red')
# ax1.step(x, sl_hist.values(), alpha=0.9, label=r"tt: sl decay", color='green')
# ax1.step(x, dl_hist.values(), alpha=0.9, label=r"tt: dl decay", color='blue')
# ax1.step(x, fh_hist.values(), alpha=0.9, label=r"tt: fh decay", color='tab:brown')
# ax1.step(x, dy_hist.values(), alpha=0.9, label=r"dy", color='tab:pink')


# ax1.tick_params(axis='y', labelcolor='black')

# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = '#4b2e83'
# ax2.set_ylabel(r'asimov significance', color=color)  # we already handled the x-label with ax1
# ax2.plot(x, sig_all, label='significance', color=color, alpha=1.0)
# ax2.tick_params(axis='y', labelcolor=color)
# # ax2.set_xticks([1e-5, 1e-4, 1e-3, 1e-2])

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.set_yscale("log")
# ax2.set_yscale("log")

# ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.36, 1))
# # plt.legend(fontsize="small")
# ax1.set_xscale("linear")
# # ax1.set_ylim(bottom=1e-1)
# # fig.tight_layout()
# plt.title(fr"HH output node for signal and tt background; reference plot from other DNN; flat-s binning; total significance: {round(total_significance, 5)}", wrap=True)
# plt.savefig(f"images/input_file_dnn_ttdy", dpi=300, bbox_inches='tight')
# plt.show()


# sl_hist.reset()
# dl_hist.reset()
# fh_hist.reset()
# dy_hist.reset()
# all_tt_hist.reset()
# hh_hist.reset()
