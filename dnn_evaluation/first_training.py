import itertools
import torch
import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt


"""This script analyses the etau channel, res2b category and matches e, tau to gen W children.
"""
n_bins = 100

delr_cut_e = 0.25
delr_cut_b = 0.4

eps = 1e-6 # set eps=0 for normal scale
lower_border = -14# set to 0 for lin scale
upper_border = 10# set to 1 for lin scale
def logit(x):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, lower_border, upper_border-eps)
def inverse_logit(y):
    x = 1 / (1 + np.exp(-y))
    return np.clip(x, eps, 1 - eps)
def identity(x):
    return x
func = logit
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
events_train_dl4 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test.pt")
events_train_dl2 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl2.pt")
events_train_dl6 = torch.load("/data/dust/user/hergesk/HH_DNN/evaluation/test_dl6.pt")
from IPython import embed; embed(header="MESSAGE Line 34 | File: first_training.py")
# events_train = events_train[events_train.run3_dnn_moe_hh > 0]
for events, label in zip([events_train_dl4, events_train_dl2, events_train_dl6], [2, 4, 6]):
# split the tt bg data in three processes and convert torch tensors to np arrays
    events_tt_dl = events[0]["test"][('tt', 1200)]
    events_tt_fh = events[0]["test"][('tt', 1300)]
    events_tt_sl = events[0]["test"][('tt', 1100)]
    events_hh    = events[0]["test"][('hh', 21101)]

    dl_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
    fh_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
    sl_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
    hh_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
    all_tt_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
    # fill
    dl_hist.fill(func(events_tt_dl["scores"].numpy()[:, 0]), weight =events_tt_dl["event_weight"].numpy())
    fh_hist.fill(func(events_tt_fh["scores"].numpy()[:, 0]), weight =events_tt_fh["event_weight"].numpy())
    sl_hist.fill(func(events_tt_sl["scores"].numpy()[:, 0]), weight =events_tt_sl["event_weight"].numpy())
    hh_hist.fill(func(events_hh["scores"].numpy()[:, 0]), weight =events_hh["event_weight"].numpy())
    all_tt_hist.fill(func(events_tt_dl["scores"].numpy()[:, 0]), weight =events_tt_dl["event_weight"].numpy())
    all_tt_hist.fill(func(events_tt_fh["scores"].numpy()[:, 0]), weight =events_tt_fh["event_weight"].numpy())
    all_tt_hist.fill(func(events_tt_sl["scores"].numpy()[:, 0]), weight =events_tt_sl["event_weight"].numpy())


    scaling_factor = ((hh_hist.values().sum())/ (all_tt_hist.values().sum()))**(-1)
    # plot
    x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.set_xlabel('HH output node')
    ax1.set_ylabel('Number of events', color="black")
    ax1.step(x, sl_hist.values(), alpha=0.9, label=r"tt: sl decay", color='green')
    ax1.step(x, dl_hist.values(), alpha=0.9, label=r"tt: dl decay", color='blue')
    ax1.step(x, fh_hist.values(), alpha=0.9, label=r"tt: fh decay", color='tab:pink')
    ax1.step(x, all_tt_hist.values(), alpha=0.9, label=r"tt: all events", color='red')
    ax1.step(x, hh_hist.values()* scaling_factor, alpha=0.9, label=fr"signal x {round(scaling_factor)}", color="black")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_legend_handles_labels()
    plt.legend(fontsize="small")
    ax1.set_yscale("log")
    ax1.set_xscale("linear")
    ax1.set_ylim(bottom=1e-1)
    fig.tight_layout()
    plt.title(fr"HH output node for signal and tt background; sampled: (fh: 1; sl: 1; dl: {label})", wrap=True)
    plt.savefig(f"images/first_training_11{label}", dpi=300, bbox_inches='tight')
    plt.show()


    sl_hist.reset()
    dl_hist.reset()
    fh_hist.reset()
    all_tt_hist.reset()
    hh_hist.reset()
