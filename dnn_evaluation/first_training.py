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
from modules import logit, identity, asimov_significance, flats_binning, add_flow_bin

"""This script analyses the first trained DNN's which sample the tt background in different ways concerning their W decay mode,
meaning di-leptonic, semi-leptonic and full-hadronic W decay, for a flat-s binning. The Asimov significance is computed.
"""
n_bins = 10
eps = 1e-6 # set eps=0 for normal scale
lower_border = -1e2# set to 0 for lin scale
# upper_border = 12# set to 1 for lin scale
func = identity

label_color = '#4b2e83'
colors = [
"#F28EBC",  # light pink
"#D45087",  # rose
"#9B5DE5",  # lavender violet
"#5A189A",  # dark violet
]

# to get better error messages:
np.seterr(invalid="raise")

# my oversampling dl DNNs
events_reference = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/referenz_dl1.pt", map_location=torch.device('cpu'))
events_train_dl2 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl2.pt", map_location=torch.device('cpu'))
events_train_dl4 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl4.pt", map_location=torch.device('cpu'))
events_train_dl6 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl6.pt", map_location=torch.device('cpu'))

# my oversampling sl DNNs
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
    for dataset in ["training", "validation", "test"]:
        # split the tt bg data in three processes
        events_tt_dl = events[0][dataset][('tt', 1200)]
        events_tt_fh = events[0][dataset][('tt', 1300)]
        events_tt_sl = events[0][dataset][('tt', 1100)]
        events_hh    = events[0][dataset][('hh', 21101)] # signal for kappa lambda = 1, kappa t = 1
        # concatenate all dy events, which currently are stored as a dict:
        dy_indices = [k[1] for k in events[0][dataset].keys() if k[0] == "dy"]
        dicts = [(events[0][dataset][('dy', i)]) for i in dy_indices]
        # TODO: code is error-prone as new columns added to the NN output will not be adopted immediately
        events_dy = {
            "scores": torch.cat([d["scores"] for d in dicts]),
            "event_weight": torch.cat([d["event_weight"] for d in dicts]),
            "normalization_weights": torch.cat([d["normalization_weights"] for d in dicts]),
            "event_id": torch.cat([d["event_id"] for d in dicts]),
        }
        # get bin edges for flat s binning
        bin_edges = flats_binning(events_hh["scores"][:, 0], bin_num = n_bins, hist_edge_l=lower_border)[2]
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
        dl_hist = Hist(hist.axis.Variable(bin_edges, name="dl_hist", label=r"", flow=True), storage=hist.storage.Weight())
        fh_hist = Hist(hist.axis.Variable(bin_edges, name="fh_hist", label=r"", flow=True), storage=hist.storage.Weight())
        sl_hist = Hist(hist.axis.Variable(bin_edges, name="sl_hist", label=r"", flow=True), storage=hist.storage.Weight())
        dy_hist = Hist(hist.axis.Variable(bin_edges, name="dy_hist", label=r"", flow=True), storage=hist.storage.Weight())
        hh_hist = Hist(hist.axis.Variable(bin_edges, name="hh_hist", label=r"", flow=True), storage=hist.storage.Weight())
        all_tt_hist = Hist(hist.axis.Variable(bin_edges, name="all_tt_hist", label=r"", flow=True), storage=hist.storage.Weight())
        # fill
        dl_hist.fill(func(events_tt_dl["scores"].numpy()[:, 0]), weight =events_tt_dl["event_weight"].numpy()* events_tt_dl["normalization_weights"].numpy())
        fh_hist.fill(func(events_tt_fh["scores"].numpy()[:, 0]), weight =events_tt_fh["event_weight"].numpy()* events_tt_fh["normalization_weights"].numpy())
        sl_hist.fill(func(events_tt_sl["scores"].numpy()[:, 0]), weight =events_tt_sl["event_weight"].numpy()* events_tt_sl["normalization_weights"].numpy())
        dy_hist.fill(func(events_dy["scores"].numpy()[:, 0]), weight =events_dy["event_weight"].numpy()* events_dy["normalization_weights"].numpy())
        hh_hist.fill(func(events_hh["scores"].numpy()[:, 0]), weight =events_hh["event_weight"].numpy()* events_hh["normalization_weights"].numpy())
        all_tt_hist.fill(func(events_tt_dl["scores"].numpy()[:, 0]), weight =events_tt_dl["event_weight"].numpy()* events_tt_dl["normalization_weights"].numpy())
        all_tt_hist.fill(func(events_tt_fh["scores"].numpy()[:, 0]), weight =events_tt_fh["event_weight"].numpy()* events_tt_fh["normalization_weights"].numpy())
        all_tt_hist.fill(func(events_tt_sl["scores"].numpy()[:, 0]), weight =events_tt_sl["event_weight"].numpy()* events_tt_sl["normalization_weights"].numpy())

        sig_all, error_sig_all = asimov_significance(hh_hist, dy_hist, fh_hist, dl_hist, sl_hist, error_type="poisson_weighted")
        sig_dl, error_sig_dl = asimov_significance(hh_hist, dl_hist, error_type="poisson_weighted")
        sig_sl, error_sig_sl = asimov_significance(hh_hist, sl_hist, error_type="poisson_weighted")
        sig_fh, error_sig_fh = asimov_significance(hh_hist, fh_hist, error_type="poisson_weighted")
        all_significances = [sig_all, sig_dl, sig_sl, sig_fh]
        all_errors = [error_sig_all, error_sig_dl, error_sig_sl, error_sig_fh]
        all_sig_tot = [np.sqrt(np.sum(np.square(s))) for s in all_significances]
        scaling_factor = ((add_flow_bin(hh_hist).sum())/ (add_flow_bin(all_tt_hist).sum() + add_flow_bin(dy_hist).sum()))**(-1)
        # scaling_factor = ((hh_hist.values().sum())/ (all_tt_hist.values().sum() ))**(-1)
        # plot
        x = bin_edges  # bin edges
        x_bin_centers = (x[:-1] + x[1:]) / 2  # bin centers
        x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
        x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers
        fig, ax1 = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(right=0.85)
        ax1.stairs(add_flow_bin(all_tt_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='red', label=r"tt: all events")
        ax1.stairs(add_flow_bin(sl_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='green', label=r"tt: sl decay")
        ax1.stairs(add_flow_bin(dl_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='blue', label=r"tt: dl decay")
        ax1.stairs(add_flow_bin(fh_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:brown', label=r"tt: fh decay")
        ax1.stairs(add_flow_bin(dy_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:orange', label=r"dy")
        ax1.stairs(add_flow_bin(hh_hist)*scaling_factor, edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor="black", label=fr"signal x {round(scaling_factor)}")

        ax1.tick_params(axis='y', labelcolor='black')

        yaxis_sig = ax1.twinx()
        yaxis_sig.set_ylabel(r'asimov significance $Z_A$', color=label_color)
        for sig, error, sig_tot, label, color in zip(all_significances,
                                                all_errors,
                                                all_sig_tot,
                                                [r"$Z_A$: tt+dy", r"$Z_A$: tt dl", r"$Z_A$: tt sl", r"$Z_A$: tt fh"],
                                                colors):
            yaxis_sig.errorbar(x_lin_bincenters, sig, yerr=error, label=label+f"; total: {round(sig_tot, 2)}", color=color, alpha=1.0, elinewidth=0.5, capsize=2)# , errorevery=2
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
        ax1_upper.set_xticklabels(range(1,len(x_lin_binedges)), rotation=0)  # Set text labels.
        ax1_upper.set_xlabel('bin number')

        yaxis_sig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.45, 1))
        # plt.legend(fontsize="small")
        # ax1.set_ylim(bottom=1e-1)
        # fig.tight_layout()
        plt.title(fr"HH output node for signal ($\kappa_\lambda = 1, \kappa_t = 1$) and tt background; {label1}; {dataset} data; flat-s binning; total Asimov significance: $Z_A$ = {round(all_sig_tot[0], 5)}", wrap=True, pad=13)
        plt.savefig(f"images/first_training_{label2}_{dataset}_ttdy", dpi=300, bbox_inches='tight')
        plt.show()


        sl_hist.reset()
        dl_hist.reset()
        fh_hist.reset()
        dy_hist.reset()
        all_tt_hist.reset()
        hh_hist.reset()

################################################################################################
################################################################################################
################################################################################################
# sanity-check: "old data"
func = logit
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/tt_22pre_v14.parquet")
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/hh_22pre_v14.parquet")
events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/dy_22pre_v14.parquet")

events_hh = events_hh[events_hh.run3_dnn_moe_hh > 0]
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_dy = events_dy[events_dy.run3_dnn_moe_hh > 0]

events_hh = events_hh[events_hh.process_id == 21101] # signal for kappa lambda = 1, kappa t = 1
events_tt_dl = events_tt[events_tt.process_id == 1200]
events_tt_fh = events_tt[events_tt.process_id == 1300]
events_tt_sl = events_tt[events_tt.process_id == 1100]
convert_to_logit = lambda x: func(x.run3_dnn_moe_hh)
from IPython import embed; embed(header="MESSAGE Line 182 | File: first_training.py")
# for i in [events_tt_dl, events_tt_fh, events_tt_sl, events_hh, events_dy]:
#     i = convert_to_logit(i)
# get bin edges for flat s binning
bin_edges = flats_binning(torch.from_numpy(ak.to_numpy(convert_to_logit(events_hh))), bin_num = n_bins, hist_edge_l=lower_border)[2]
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

dl_hist = Hist(hist.axis.Variable(bin_edges, name="dl_hist", label=r"", flow=True), storage=hist.storage.Weight())
fh_hist = Hist(hist.axis.Variable(bin_edges, name="fh_hist", label=r"", flow=True), storage=hist.storage.Weight())
sl_hist = Hist(hist.axis.Variable(bin_edges, name="sl_hist", label=r"", flow=True), storage=hist.storage.Weight())
dy_hist = Hist(hist.axis.Variable(bin_edges, name="dy_hist", label=r"", flow=True), storage=hist.storage.Weight())
hh_hist = Hist(hist.axis.Variable(bin_edges, name="hh_hist", label=r"", flow=True), storage=hist.storage.Weight())
all_tt_hist = Hist(hist.axis.Variable(bin_edges, name="all_tt_hist", label=r"", flow=True), storage=hist.storage.Weight())
# fill
dl_hist.fill(convert_to_logit(events_tt_dl), weight =events_tt_dl.event_weight)
fh_hist.fill(convert_to_logit(events_tt_fh), weight =events_tt_fh.event_weight)
sl_hist.fill(convert_to_logit(events_tt_sl), weight =events_tt_sl.event_weight)
dy_hist.fill(convert_to_logit(events_dy), weight =events_dy.event_weight)
hh_hist.fill(convert_to_logit(events_hh), weight =events_hh.event_weight)
all_tt_hist.fill(convert_to_logit(events_tt_dl), weight =events_tt_dl.event_weight)
all_tt_hist.fill(convert_to_logit(events_tt_fh), weight =events_tt_fh.event_weight)
all_tt_hist.fill(convert_to_logit(events_tt_sl), weight =events_tt_sl.event_weight)

sig_all, error_sig_all = asimov_significance(hh_hist, dy_hist, fh_hist, dl_hist, sl_hist, error_type="poisson_weighted")
sig_dl, error_sig_dl = asimov_significance(hh_hist, dl_hist, error_type="poisson_weighted")
sig_sl, error_sig_sl = asimov_significance(hh_hist, sl_hist, error_type="poisson_weighted")
sig_fh, error_sig_fh = asimov_significance(hh_hist, fh_hist, error_type="poisson_weighted")

all_significances = [sig_all, sig_dl, sig_sl, sig_fh]
all_errors = [error_sig_all, error_sig_dl, error_sig_sl, error_sig_fh]
all_sig_tot = [np.sqrt(np.sum(np.square(s))) for s in all_significances]
scaling_factor = ((add_flow_bin(hh_hist).sum())/ (add_flow_bin(all_tt_hist).sum() + add_flow_bin(dy_hist).sum()))**(-1)
# scaling_factor = ((hh_hist.values().sum())/ (all_tt_hist.values().sum() ))**(-1)
# plot
x = bin_edges  # bin edges
x_bin_centers = (x[:-1] + x[1:]) / 2  # bin centers
x_lin_binedges = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x_lin_bincenters = (x_lin_binedges[:-1] + x_lin_binedges[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.stairs(add_flow_bin(hh_hist)*scaling_factor, edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor="black", label=fr"signal x {round(scaling_factor)}")
ax1.stairs(add_flow_bin(all_tt_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='red', label=r"tt: all events")
ax1.stairs(add_flow_bin(sl_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='green', label=r"tt: sl decay")
ax1.stairs(add_flow_bin(dl_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='blue', label=r"tt: dl decay")
ax1.stairs(add_flow_bin(fh_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:brown', label=r"tt: fh decay")
ax1.stairs(add_flow_bin(dy_hist), edges = x_lin_binedges, linewidth=1.5, baseline=0, fill=False, edgecolor='tab:orange', label=r"dy")
ax1.tick_params(axis='y', labelcolor='black')

yaxis_sig = ax1.twinx()
yaxis_sig.set_ylabel(r'asimov significance $Z_A$', color=label_color)
for sig, error, sig_tot, label, color in zip(all_significances,
                                        all_errors,
                                        all_sig_tot,
                                        [r"$Z_A$: tt+dy", r"$Z_A$: tt dl", r"$Z_A$: tt sl", r"$Z_A$: tt fh"],
                                        colors):
    yaxis_sig.errorbar(x_lin_bincenters, sig, yerr=error, label=label+f"; total: {round(sig_tot, 2)}", color=color, alpha=1.0, elinewidth=0.5, capsize=2)# , errorevery=2
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
ax1_upper.set_xticklabels(range(1,len(x_lin_binedges)), rotation=0)  # Set text labels.
ax1_upper.set_xlabel('bin number')

yaxis_sig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.45, 1))
# plt.legend(fontsize="small")
# ax1.set_ylim(bottom=1e-1)
# fig.tight_layout()
plt.title(fr"HH output node for signal ($\kappa_\lambda = 1, \kappa_t = 1$) and tt background; original input file; flat-s binning; total Asimov significance: $Z_A$ = {round(all_sig_tot[0], 5)}", wrap=True, pad=13)
plt.savefig(f"images/input_file_ttdy", dpi=300, bbox_inches='tight')
plt.show()

sl_hist.reset()
dl_hist.reset()
fh_hist.reset()
dy_hist.reset()
all_tt_hist.reset()
hh_hist.reset()
