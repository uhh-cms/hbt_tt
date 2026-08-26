import itertools
from multiprocessing.util import debug
import torch
import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import functools
import operator
from termcolor import colored
from IPython import embed
from modules import logit, identity, asimov_significance, flats_binning, add_flow_bin
from structures import ProcessLoader, Process, HistFab

"""This script analyses the first trained DNN's which sample the tt background in different ways concerning their W decay mode,
meaning di-leptonic, semi-leptonic and full-hadronic W decay, for a flat-s binning. The Asimov significance is computed.
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

# to get better error messages:
np.seterr(invalid="raise")

path_dnn = "/data/dust/user/hergesk/HH_DNN/evaluation"
path_old_dnn = "/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24"
process_loader = ProcessLoader()

data_dnn_outputs = [
    # my DNNs oversampling dl
    process_loader.load_process(path_dnn+"/22_23_lr_linear_a_test_test.pt", label="22_23_lr_linear_a_test", description="Weighting: (1,1,1); lin LR"),
    process_loader.load_process(path_dnn+"/22_23_lr_lin_plus_cos_b_test.pt", label="22_23_lr_lin_plus_cos_b_test", description=r"Weighting: (1,1,1); LR = $10^{-3}$"),
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
    # TODO implement an etau, mutau, tautau split in a manner such that the structures.py classes are still applicable
    legend_handles = []
    legend_labels = []
    split_events = [
        output.split_into_categories("etau"),
        output.split_into_categories("mutau"),
        output.split_into_categories("tautau")
    ]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, d, small_title in zip(axs, split_events, ["etau", "mutau", "tautau"]):
        # access all events
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
        sig_all, error_sig_all = asimov_significance(histograms["hh_hist"], histograms["dy_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")
        sig_dl, error_sig_dl = asimov_significance(histograms["hh_hist"], histograms["dl_hist"], error_type="poisson_weighted")
        sig_sl, error_sig_sl = asimov_significance(histograms["hh_hist"], histograms["sl_hist"], error_type="poisson_weighted")
        sig_fh, error_sig_fh = asimov_significance(histograms["hh_hist"], histograms["fh_hist"], error_type="poisson_weighted")
        all_significances = [sig_all, sig_dl, sig_sl, sig_fh]
        all_errors = [error_sig_all, error_sig_dl, error_sig_sl, error_sig_fh]
        all_sig_tot = [np.sqrt(np.sum(np.square(s))) for s in all_significances]
        scaling_factor = ((add_flow_bin(histograms["hh_hist"]).sum()+eps)/ (add_flow_bin(histograms["all_tt_hist"]).sum() + add_flow_bin(histograms["dy_hist"]).sum())+eps)**(-1)
        x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
        x = (x[:-1] + x[1:]) / 2  # bin centers
        # ---
        x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
        x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers
        # fig, ax1 = plt.subplots(figsize=(7, 5))
        # fig.subplots_adjust(right=0.85)
        # TODO: loop through hists instead of this error-prone stuff I'm doing here
        # ax1.stairs(add_flow_bin(histograms["all_tt_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='red', label=r"tt: all events")
        ax.stairs(add_flow_bin(histograms["sl_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor=hists[1].color, label=r"tt: sl decay")
        ax.stairs(add_flow_bin(histograms["dl_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor=hists[2].color, label=r"tt: dl decay")
        ax.stairs(add_flow_bin(histograms["fh_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor=hists[3].color, label=r"tt: fh decay")
        ax.stairs(add_flow_bin(histograms["hh_hist"])*scaling_factor, edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor=hists[5].color, label=fr"signal x scaling factor")
        ax.stairs(add_flow_bin(histograms["dy_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, linestyle="--", fill=False, edgecolor=hists[4].color, label=r"dy")

        ax.tick_params(axis='y', labelcolor='black')

        yaxis_sig = ax.twinx()
        yaxis_sig.set_ylabel(r"$\mathbf{Z_A}$", labelpad=4, color=label_color)
        yaxis_sig.yaxis.set_label_coords(1.04, 0.94)
        # for sig, error, sig_tot, label, color in zip(all_significances,
        #                                         all_errors,
        #                                         all_sig_tot,
        #                                         [r"$Z_A$: tt+dy", r"$Z_A$: tt dl", r"$Z_A$: tt sl", r"$Z_A$: tt fh"],
        #                                         colors):
        #     yaxis_sig.errorbar(x_lin_bincenters, sig, yerr=error, label=label+f"; total: {round(sig_tot, 2)}", color=color, alpha=1.0, elinewidth=0.5, capsize=2)# , errorevery=2
        yaxis_sig.errorbar(x_lin_bincenters, sig_all, yerr=error_sig_all, label=fr"$Z_A$; total: {round(all_sig_tot[0], 2)}", color=label_color, alpha=1.0, elinewidth=0.5, capsize=2)# , errorevery=2
        yaxis_sig.tick_params(axis='y', labelcolor=label_color)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = yaxis_sig.get_legend_handles_labels()

        if not legend_handles:
            legend_handles = lines1 + lines2
            legend_labels = labels1 + labels2
            
        ax.set_yscale("log")
        yaxis_sig.set_yscale("log")
        ax.set_ylabel(r"Events", labelpad=4)
        ax.yaxis.set_label_coords(-0.08, 0.94)
        # lower x axis with bin edges
        ax.set_xticks(x_lin_binedges)  # Set label locations.
        ax.set_xticklabels(x_lin_binedges.astype(int), rotation=45)  # Set text labels.
        ax.set_xlabel('HH output node')
        

        # upper x axis with bin numbers
        # ax_upper = ax.twiny()
        # ax_upper.set_xlim(ax.get_xlim())
        # ax_upper.set_xticks(x_lin_bincenters)
        # ax_upper.set_xticklabels(range(1,len(x_lin_binedges)), rotation=0)  # Set text labels.
        # ax_upper.set_xlabel('bin number')
        ax.set_title(small_title + ", res2b", fontsize=11, pad=10)
        for h in histograms:
            histograms[h].reset()
        # yaxis_sig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.38, 1))
    # plt.legend(fontsize="small")
    # ax1.set_ylim(bottom=1e-1)
    # fig.tight_layout()
    # fig.legend(
    #     legend_handles,
    #     legend_labels,
    #     loc="center right",
    #     bbox_to_anchor=(1.1, 0.5)
    # )
    fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=len(legend_labels),
    )
    fig.suptitle(fr"HH output node for signal ($\kappa_\lambda = 1, \kappa_t = 1$) and background; {d.description}; test data", ha="center", fontweight="bold", y=1.12)
    plt.savefig(f"images_ref/first_training_{d.label}_ttdy", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


################################################################################################
################################################################################################
################################################################################################
# loop without flat-s binning:
# lower_border = -14
# upper_border = 11

# for d in data_dnn_outputs:
#     print(f"Processing label {d.label}")
#     for dataset in ["training", "validation", "test"]:
#         print("processing dataset: ", dataset)
#         # access all events
#         # split the tt bg data in three processes
#         # events_dict = d.get_events(dataset)
#         hists = [HistFab("all_tt_hist", ["tt_dl", "tt_sl", "tt_fh"], "red", "tt: all events", flavor=d.flavor),
#                  HistFab("sl_hist", ["tt_sl"], 'green', "tt: sl events", flavor=d.flavor),
#                  HistFab("dl_hist", ["tt_dl"], 'blue', "tt: dl events", flavor=d.flavor),
#                  HistFab("fh_hist", ["tt_fh"], 'tab:brown', "tt: fh events", flavor=d.flavor),
#                  HistFab("dy_hist", ["dy"], 'tab:orange', "dy: all events", flavor=d.flavor),
#                  HistFab("hh_hist", ["hh"], "black", "hh: all events", flavor=d.flavor)
#         ]

#         histograms = {}
#         for h in hists:
#             histogram = h.create_hist(n_bins, lower_border, upper_border)
#             h.fill_hist(
#                 histogram,
#                 func,
#                 d)
#             histograms[h.name] = histogram
#         sig_all, error_sig_all = asimov_significance(histograms["hh_hist"], histograms["dy_hist"], histograms["fh_hist"], histograms["dl_hist"], histograms["sl_hist"], error_type="poisson_weighted")
#         sig_dl, error_sig_dl = asimov_significance(histograms["hh_hist"], histograms["dl_hist"], error_type="poisson_weighted")
#         sig_sl, error_sig_sl = asimov_significance(histograms["hh_hist"], histograms["sl_hist"], error_type="poisson_weighted")
#         sig_fh, error_sig_fh = asimov_significance(histograms["hh_hist"], histograms["fh_hist"], error_type="poisson_weighted")
#         all_significances = [sig_all, sig_dl, sig_sl, sig_fh]
#         all_errors = [error_sig_all, error_sig_dl, error_sig_sl, error_sig_fh]
#         all_sig_tot = [np.sqrt(np.sum(np.square(s))) for s in all_significances]
#         scaling_factor = ((add_flow_bin(histograms["hh_hist"]).sum()+eps)/ (add_flow_bin(histograms["all_tt_hist"]).sum() + add_flow_bin(histograms["dy_hist"]).sum())+eps)**(-1)

#         x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
#         x = (x[:-1] + x[1:]) / 2  # bin centers
#         # ---
#         x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
#         x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers
#         fig, ax1 = plt.subplots(figsize=(9, 5))
#         fig.subplots_adjust(right=0.85)
#         ax1.stairs(add_flow_bin(histograms["all_tt_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='red', label=r"tt: all events")
#         ax1.stairs(add_flow_bin(histograms["sl_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='green', label=r"tt: sl decay")
#         ax1.stairs(add_flow_bin(histograms["dl_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='blue', label=r"tt: dl decay")
#         ax1.stairs(add_flow_bin(histograms["fh_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:brown', label=r"tt: fh decay")
#         ax1.stairs(add_flow_bin(histograms["dy_hist"]), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:orange', label=r"dy")
#         ax1.stairs(add_flow_bin(histograms["hh_hist"])*scaling_factor, edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor="black", label=fr"signal x {round(scaling_factor)}")

#         ax1.tick_params(axis='y', labelcolor='black')

#         yaxis_sig = ax1.twinx()
#         yaxis_sig.set_ylabel(r'asimov significance $Z_A$', color=label_color)
#         for sig, error, sig_tot, label, color in zip(all_significances,
#                                                 all_errors,
#                                                 all_sig_tot,
#                                                 [r"$Z_A$: tt+dy", r"$Z_A$: tt dl", r"$Z_A$: tt sl", r"$Z_A$: tt fh"],
#                                                 colors):
#             yaxis_sig.errorbar(x_lin_bincenters, sig, yerr=error, label=label+f"; total: {round(sig_tot, 2)}", color=color, alpha=1.0, elinewidth=0.5, capsize=2)# , errorevery=2
#         yaxis_sig.tick_params(axis='y', labelcolor=label_color)
#         lines1, labels1 = ax1.get_legend_handles_labels()
#         lines2, labels2 = yaxis_sig.get_legend_handles_labels()
#         ax1.set_yscale("log")
#         yaxis_sig.set_yscale("log")
#         ax1.set_ylabel('Number of events', color="black")
#         # lower x axis with bin edges
#         ax1.set_xticks(x_lin_binedges)  # Set label locations.
#         ax1.set_xticklabels(x_lin_binedges.round(2), rotation=45)  # Set text labels.
#         ax1.set_xlabel('HH output node')

#         # upper x axis with bin numbers
#         ax1_upper = ax1.twiny()
#         ax1_upper.set_xlim(ax1.get_xlim())
#         ax1_upper.set_xticks(x_lin_bincenters)
#         ax1_upper.set_xticklabels(range(1,len(x_lin_binedges)), rotation=0)  # Set text labels.
#         ax1_upper.set_xlabel('bin number')

#         yaxis_sig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.45, 1))
#         # plt.legend(fontsize="small")
#         # ax1.set_ylim(bottom=1e-1)
#         # fig.tight_layout()
#         plt.title(fr"HH output node for signal ($\kappa_\lambda = 1, \kappa_t = 1$) and background; {d.description}; {dataset} data; total Asimov significance: $Z_A$ = {round(all_sig_tot[0], 5)}", wrap=True, pad=13)
#         plt.savefig(f"images_ref/first_training_{d.label}_{dataset}_ttdy_regularbinning", dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
