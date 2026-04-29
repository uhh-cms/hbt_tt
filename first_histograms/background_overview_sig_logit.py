import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script computes and plots the significance of HH signal and (tt and dy)
background data in the HH output node with logit x-scale.
"""
events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data


n_bins = 60

eps = 1e-6 # set eps=0 for normal scale
def logit(x):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, -14, 5-eps)
    #return x
# discard negative values to avoid errors in logit transformation
events_hh = events_hh[events_hh.run3_dnn_moe_hh > 0]
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_dy = events_dy[events_dy.run3_dnn_moe_hh > 0]

# initialize histograms
hh = Hist(hist.axis.Regular(n_bins, logit(eps), 5, name="hh", label="hh"))
tt = Hist(hist.axis.Regular(n_bins, logit(eps), 5, name="tt", label="tt"))
dy = Hist(hist.axis.Regular(n_bins, logit(eps), 5, name="dy", label="dy"))

# first plot: plot all tt background together
# fill histograms
hh.fill(logit(events_hh.run3_dnn_moe_hh), weight =events_hh.event_weight)
tt.fill(logit(events_tt.run3_dnn_moe_hh), weight =events_tt.event_weight)
dy.fill(logit(events_dy.run3_dnn_moe_hh), weight =events_dy.event_weight)

# significance
def significance(s, *b):
    """
    Computes the significance, signal squared over background,
    per bin, for the number of bins defined above as n_bins"""
    s_count = s.values()
    b_count = np.sum([_b.values() for _b in b], axis=0)

    sig_per_bin = s_count**2 / b_count
    return sig_per_bin


sig = significance(hh, tt, dy)
total_significance = np.sqrt(np.sum(np.square(sig)))


# plot histograms
x = np.linspace(-14, 5, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
# fig = plt.figure(figsize=(10, 6))

# scale the hh histogram up, weighted by the integral of the dy and tt data
scaling_factor = (hh.values().sum() / (tt.values().sum() + dy.values().sum()))**(-1)

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)

color = 'black'
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color=color)
ax1.bar(x, tt.values(), width=(logit(eps)-logit(1-eps))/n_bins, bottom=None, alpha=0.5, label='tt', color='violet', edgecolor='black')
ax1.bar(x, dy.values(), width=(logit(eps)-logit(1-eps))/n_bins, bottom=tt.values(), alpha=0.5, label='dy', color='red', edgecolor='black')
ax1.bar(x, hh.values() * scaling_factor, width=(logit(eps)-logit(1-eps))/n_bins, bottom=None, fill=False, label=f'hh x ({scaling_factor:.2f})', color='green', edgecolor='black')

ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = '#4b2e83'
ax2.set_ylabel('significance', color=color)  # we already handled the x-label with ax1
ax2.plot(x, sig, label='significance', color=color, alpha=1.0)
ax2.tick_params(axis='y', labelcolor=color)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.set_yscale("log")
ax2.set_yscale("log")
fig.tight_layout()


ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.45, 1))

plt.title(f"logit of HH output node; total significance = {round(total_significance, 2)}")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


plt.savefig("images_hists/hh_output_node_logscale_sig_logit.png", dpi=300, bbox_inches='tight')
#plt.show()
tt.reset()

# subsequent plots: further split tt bg
# channel and process id masks
# fh, sl, dl: fully hadronic, semileptonic, di-leptonic
etau_mask_sl = (events_tt.channel_id == 1) & (events_tt.process_id == 1100)
etau_mask_dl = (events_tt.channel_id == 1) & (events_tt.process_id == 1200)
etau_mask_fh = (events_tt.channel_id == 1) & (events_tt.process_id == 1300)

mutau_mask_sl = (events_tt.channel_id == 2) & (events_tt.process_id == 1100)
mutau_mask_dl = (events_tt.channel_id == 2) & (events_tt.process_id == 1200)
mutau_mask_fh = (events_tt.channel_id == 2) & (events_tt.process_id == 1300)

tautau_mask_sl = (events_tt.channel_id == 3) & (events_tt.process_id == 1100)
tautau_mask_dl = (events_tt.channel_id == 3) & (events_tt.process_id == 1200)
tautau_mask_fh = (events_tt.channel_id == 3) & (events_tt.process_id == 1300)

# prepare plotting loop
masks = [[etau_mask_sl, etau_mask_dl, etau_mask_fh], [mutau_mask_sl, mutau_mask_dl, mutau_mask_fh], [tautau_mask_sl, tautau_mask_dl, tautau_mask_fh]]
labels = ["etau", "mutau", "tautau"]


for mask, label in zip(masks, labels):
    # initialize histograms
    tt_sl =   Hist(hist.axis.Regular(n_bins, logit(eps), 5, name="tt", label="tt"))
    tt_dl =  Hist(hist.axis.Regular(n_bins, logit(eps), 5, name="tt", label="tt"))
    tt_fh = Hist(hist.axis.Regular(n_bins, logit(eps), 5, name="tt", label="tt"))
    # fill histograms
    tt_sl.fill(logit(events_tt.run3_dnn_moe_hh[mask[0]]), weight =events_tt.event_weight[mask[0]])
    tt_dl.fill(logit(events_tt.run3_dnn_moe_hh[mask[1]]), weight =events_tt.event_weight[mask[1]])
    tt_fh.fill(logit(events_tt.run3_dnn_moe_hh[mask[2]]), weight =events_tt.event_weight[mask[2]])
    # plot
    significance_cat = significance(hh, tt_sl, tt_dl, tt_fh, dy)
    total_significance_cat = np.sqrt(np.sum(np.square(significance_cat)))

    # quick check bc significance looks very similar in every plot
    # print(label,
    #   ak.sum(mask[0]),
    #   ak.sum(mask[1]),
    #   ak.sum(mask[2]))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    color = 'black'
    bottom = np.zeros_like(x)
    ax1.bar(x, tt_sl.values(), width=(logit(eps)-logit(1-eps))/n_bins, bottom=bottom, alpha=0.5, label='tt semi-leptonic',  edgecolor='black')
    bottom+=tt_sl.values()
    ax1.bar(x, tt_dl.values(), width=(logit(eps)-logit(1-eps))/n_bins, bottom=bottom, alpha=0.5, label='tt di-leptonic',  edgecolor='black')
    bottom+=tt_dl.values()
    ax1.bar(x, tt_fh.values(), width=(logit(eps)-logit(1-eps))/n_bins, bottom=bottom, alpha=0.5, label='tt fully hadronic', edgecolor='black')
    bottom+=tt_fh.values()
    ax1.bar(x, dy.values(), width=(logit(eps)-logit(1-eps))/n_bins, bottom=bottom, alpha=0.5, label='dy', color='red', edgecolor='black')
    ax1.bar(x, hh.values() * scaling_factor, width=(logit(eps)-logit(1-eps))/n_bins, bottom=None, fill=False, label=f'hh x ({scaling_factor:.2f})', color='green', edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel("HH output node")
    ax1.set_ylabel("Number of events")

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = '#4b2e83'
    ax2.set_ylabel('significance', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, significance_cat, label='significance', color=color, alpha=1.0)

    ax2.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.45, 1))
    #ax2.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    # fig.tight_layout()
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    fig.tight_layout()
    plt.title(f"HH output node for signal and background data, {label} channel; total significance = {round(total_significance_cat, 2)}")
    plt.savefig(f"images_hists/hh_output_node_histogram_{label}_logscale_sig_logit.png", dpi=300, bbox_inches='tight')
    plt.show()
    tt.reset()
    significance_cat = 0
    total_significance_cat = 0


