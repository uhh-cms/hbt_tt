import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script plots HH signal, tt and dy background data in the HH output node,
with further splitting the tt background into its three different channels ids.
"""
events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data


n_bins = 100

# initialize histograms
hh = Hist(hist.axis.Regular(n_bins, 0, 1, name="hh", label="hh"))
tt = Hist(hist.axis.Regular(n_bins, 0, 1, name="tt", label="tt"))
dy = Hist(hist.axis.Regular(n_bins, 0, 1, name="dy", label="dy"))

# first plot: plot all tt background together
# fill histograms
hh.fill(events_hh.run3_dnn_moe_hh, weight =events_hh.event_weight)
tt.fill(events_tt.run3_dnn_moe_hh, weight =events_tt.event_weight)
dy.fill(events_dy.run3_dnn_moe_hh, weight =events_dy.event_weight)

# plot histograms
x = np.linspace(0, 1, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig = plt.figure(figsize=(10, 6))

# scale the hh histogram up, weighted by the integral of the dy and tt data
scaling_factor = (hh.values().sum() / (tt.values().sum() + dy.values().sum()))**(-1)

plt.bar(x, tt.values(), width=1/n_bins, bottom=None, alpha=0.5, label='tt', color='blue', edgecolor='black')
plt.bar(x, dy.values(), width=1/n_bins, bottom=tt.values(), alpha=0.5, label='dy', color='red', edgecolor='black')
plt.bar(x, hh.values() * scaling_factor, width=1/n_bins, bottom=None, fill=False, label=f'hh x ({scaling_factor:.2f})', color='green', edgecolor='black')
plt.gca().set_yscale('log')  # Set y-axis to logarithmic scale


plt.xlabel("HH output node")
plt.ylabel("Number of events")
plt.title("HH output node for signal and background data")
plt.legend(loc = "upper center")

plt.savefig("hh_output_node_histogram.png", dpi=300, bbox_inches='tight')
plt.show()
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
    for scale in ('linear', 'log'):
        # initialize histograms
        tt_sl =   Hist(hist.axis.Regular(n_bins, 0, 1, name="tt", label="tt"))
        tt_dl =  Hist(hist.axis.Regular(n_bins, 0, 1, name="tt", label="tt"))
        tt_fh = Hist(hist.axis.Regular(n_bins, 0, 1, name="tt", label="tt"))
        # fill histograms
        tt_sl.fill(events_tt.run3_dnn_moe_hh[mask[0]], weight =events_tt.event_weight[mask[0]])
        tt_dl.fill(events_tt.run3_dnn_moe_hh[mask[1]], weight =events_tt.event_weight[mask[1]])
        tt_fh.fill(events_tt.run3_dnn_moe_hh[mask[2]], weight =events_tt.event_weight[mask[2]])
        # plot
        fig = plt.figure(figsize=(10, 6))
        bottom = np.zeros_like(x)
        plt.bar(x, tt_sl.values(), width=1/n_bins, bottom=bottom, alpha=0.5, label='tt semi-leptonic',  edgecolor='black')
        bottom+=tt_sl.values()
        plt.bar(x, tt_dl.values(), width=1/n_bins, bottom=bottom, alpha=0.5, label='tt di-leptonic',  edgecolor='black')
        bottom+=tt_dl.values()
        plt.bar(x, tt_fh.values(), width=1/n_bins, bottom=bottom, alpha=0.5, label='tt fully hadronic', edgecolor='black')
        bottom+=tt_fh.values()
        plt.bar(x, dy.values(), width=1/n_bins, bottom=bottom, alpha=0.5, label='dy', color='red', edgecolor='black')
        plt.bar(x, hh.values() * scaling_factor, width=1/n_bins, bottom=None, fill=False, label=f'hh x ({scaling_factor:.2f})', color='green', edgecolor='black')
        plt.gca().set_yscale(scale)  # Set y-axis to linear scale

        plt.xlabel("HH output node")
        plt.ylabel("Number of events")
        plt.title(f"HH output node for signal and background data, {label} channel")
        plt.legend()

        plt.savefig(f"hh_output_node_histogram_{label}_{scale}scale.png", dpi=300, bbox_inches='tight')
        plt.show()
        tt.reset()

