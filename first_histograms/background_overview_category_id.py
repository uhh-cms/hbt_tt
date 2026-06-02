import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script plots HH signal, tt and dy background data in the HH output node,
with further splitting the tt background into its six different category ids.
"""
events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data

n_bins = 100
eps = 1e-6 # set eps=0 for normal scale
eps2 = 1e-1
lower_border = -14
upper_border = 12
def logit(x):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, lower_border, upper_border-eps)

# discard negative values to avoid errors in logit transformation
events_hh = events_hh[events_hh.run3_dnn_moe_hh > 0]
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_dy = events_dy[events_dy.run3_dnn_moe_hh > 0]

# initialize and fill histograms
hh = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="hh", label="hh"))
hh.fill(logit(events_hh.run3_dnn_moe_hh), weight =events_hh.event_weight)

# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig = plt.figure(figsize=(9, 5))

def significance(s, *b):
    """
    Computes the significance, signal squared over background,
    per bin, for the number of bins defined above as n_bins"""
    s_count = s.values()
    b_count = np.sum([_b.values() for _b in b], axis=0)

    sig_per_bin = s_count**2 / (b_count + eps2)
    return np.sqrt(np.abs(sig_per_bin))

def total_significance(s):
    return np.sqrt(np.sum(np.square(s)))
# further split tt bg
# category id masks (one or two resolved b jets)
# process id: fh, sl, dl: fully hadronic, semileptonic, di-leptonic
etau_mask_res1b_sl = ak.any(events_tt.category_ids == 147, axis = 1) & (events_tt.process_id == 1100)
etau_mask_res1b_dl = ak.any(events_tt.category_ids == 147, axis = 1) & (events_tt.process_id == 1200)
etau_mask_res1b_fh = ak.any(events_tt.category_ids == 147, axis = 1) & (events_tt.process_id == 1300)
etau_mask_res1b = ak.any(events_dy.category_ids == 147, axis = 1)
etau_mask_res1b_hh = ak.any(events_hh.category_ids == 147, axis = 1)

etau_mask_res2b_sl = ak.any(events_tt.category_ids == 151, axis = 1) & (events_tt.process_id == 1100)
etau_mask_res2b_dl = ak.any(events_tt.category_ids == 151, axis = 1) & (events_tt.process_id == 1200)
etau_mask_res2b_fh = ak.any(events_tt.category_ids == 151, axis = 1) & (events_tt.process_id == 1300)
etau_mask_res2b = ak.any(events_dy.category_ids == 151, axis = 1)
etau_mask_res2b_hh = ak.any(events_hh.category_ids == 151, axis = 1)

mutau_mask_res1b_sl = ak.any(events_tt.category_ids == 175, axis = 1) & (events_tt.process_id == 1100)
mutau_mask_res1b_dl = ak.any(events_tt.category_ids == 175, axis = 1) & (events_tt.process_id == 1200)
mutau_mask_res1b_fh = ak.any(events_tt.category_ids == 175, axis = 1) & (events_tt.process_id == 1300)
mutau_mask_res1b = ak.any(events_dy.category_ids == 175, axis = 1)
mutau_mask_res1b_hh = ak.any(events_hh.category_ids == 175, axis = 1)

mutau_mask_res2b_sl = ak.any(events_tt.category_ids == 179, axis = 1) & (events_tt.process_id == 1100)
mutau_mask_res2b_dl = ak.any(events_tt.category_ids == 179, axis = 1) & (events_tt.process_id == 1200)
mutau_mask_res2b_fh = ak.any(events_tt.category_ids == 179, axis = 1) & (events_tt.process_id == 1300)
mutau_mask_res2b = ak.any(events_dy.category_ids == 179, axis = 1)
mutau_mask_res2b_hh = ak.any(events_hh.category_ids == 179, axis = 1)

tautau_mask_res1b_sl = ak.any(events_tt.category_ids == 203, axis = 1) & (events_tt.process_id == 1100)
tautau_mask_res1b_dl = ak.any(events_tt.category_ids == 203, axis = 1) & (events_tt.process_id == 1200)
tautau_mask_res1b_fh = ak.any(events_tt.category_ids == 203, axis = 1) & (events_tt.process_id == 1300)
tautau_mask_res1b = ak.any(events_dy.category_ids == 203, axis = 1)
tautau_mask_res1b_hh = ak.any(events_hh.category_ids == 203, axis = 1)

tautau_mask_res2b_sl = ak.any(events_tt.category_ids == 207, axis = 1) & (events_tt.process_id == 1100)
tautau_mask_res2b_dl = ak.any(events_tt.category_ids == 207, axis = 1) & (events_tt.process_id == 1200)
tautau_mask_res2b_fh = ak.any(events_tt.category_ids == 207, axis = 1) & (events_tt.process_id == 1300)
tautau_mask_res2b = ak.any(events_dy.category_ids == 207, axis = 1)
tautau_mask_res2b_hh = ak.any(events_hh.category_ids == 207, axis = 1)

# prepare plotting loop
masks = [[etau_mask_res1b_sl, etau_mask_res1b_dl, etau_mask_res1b_fh, etau_mask_res1b, etau_mask_res1b_hh],
         [etau_mask_res2b_sl, etau_mask_res2b_dl, etau_mask_res2b_fh, etau_mask_res2b, etau_mask_res2b_hh],
         [mutau_mask_res1b_sl, mutau_mask_res1b_dl, mutau_mask_res1b_fh, mutau_mask_res1b, mutau_mask_res1b_hh],
         [mutau_mask_res2b_sl, mutau_mask_res2b_dl, mutau_mask_res2b_fh, mutau_mask_res2b, mutau_mask_res2b_hh],
         [tautau_mask_res1b_sl, tautau_mask_res1b_dl, tautau_mask_res1b_fh, tautau_mask_res1b, tautau_mask_res1b_hh],
         [tautau_mask_res2b_sl, tautau_mask_res2b_dl, tautau_mask_res2b_fh, tautau_mask_res2b, tautau_mask_res2b_hh]]
labels = ["etau, res 1b","etau, res 2b", "mutau, res 1b", "mutau, res 2b", "tautau, res 1b", "tautau, res 2b"]
upper_borders = [10,11.5,12,12.5,10,11.5]
# upper_borders = [6, 6, 6, 6, 6.7, 6.8]
lower_xaxis_borders = [-13, -13, -13.5, -13.5, -9, -9]
hh.reset()

for mask, label, upper_border, lower_x_border in zip(masks, labels, upper_borders, lower_xaxis_borders):
    #for scale in ('linear', 'log'):
        # initialize histograms
        tt_sl =   Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="tt_sl", label="tt_sl"))
        tt_dl =   Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="tt_dl", label="tt_dl"))
        tt_fh =   Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="tt_fh", label="tt_fh"))
        dy =      Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="dy", label="dy"))
        hh =      Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="hh", label="hh"))

        # fill histograms
        tt_sl.fill(logit(events_tt.run3_dnn_moe_hh[mask[0]]), weight =events_tt.event_weight[mask[0]])
        tt_dl.fill(logit(events_tt.run3_dnn_moe_hh[mask[1]]), weight =events_tt.event_weight[mask[1]])
        tt_fh.fill(logit(events_tt.run3_dnn_moe_hh[mask[2]]), weight =events_tt.event_weight[mask[2]])
        dy.fill(logit(events_dy.run3_dnn_moe_hh[mask[3]]), weight =events_dy.event_weight[mask[3]])
        hh.fill(logit(events_hh.run3_dnn_moe_hh[mask[4]]), weight =events_hh.event_weight[mask[4]])
        # compute significance
        sig = significance(hh, tt_sl, tt_dl, tt_fh, dy)
        sig_tot = total_significance(sig)
        # scale the hh histogram up, weighted by the integral of the dy and tt data
        scaling_factor = (hh.values().sum() / (tt_sl.values().sum() + tt_dl.values().sum() + tt_fh.values().sum() + dy.values().sum()))**(-1)
        # plot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(right=0.85)
        color = 'black'
        bottom = np.zeros_like(x)
        ax1.bar(x, tt_sl.values(), width=(lower_border - upper_border)/n_bins, bottom=bottom, alpha=0.5, label='tt semi-leptonic',  edgecolor='black')
        bottom+=tt_sl.values()
        ax1.bar(x, tt_dl.values(), width=(lower_border - upper_border)/n_bins, bottom=bottom, alpha=0.5, label='tt di-leptonic',  edgecolor='black')
        bottom+=tt_dl.values()
        ax1.bar(x, tt_fh.values(), width=(lower_border - upper_border)/n_bins, bottom=bottom, alpha=0.5, label='tt fully hadronic',  edgecolor='black')
        bottom+=tt_fh.values()
        ax1.bar(x, dy.values(), width=(lower_border - upper_border)/n_bins, bottom=bottom, alpha=0.5, label='dy', color='red', edgecolor='black')
        ax1.bar(x, hh.values() * scaling_factor, width=(lower_border - upper_border)/n_bins, bottom=None, fill=False, label=f'hh x ({scaling_factor:.2f})', color='green', edgecolor='black')
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.xaxis.set_major_locator(MultipleLocator(1)) # Intervals for major x-ticks
        ax1.set_xlabel("logit of HH output node")
        ax1.set_ylabel("Number of events")
        ax1.set_ylim(bottom=1e-1)
        ax1.set_xlim(left=lower_x_border)

        ax2 = ax1.twinx()
        color = '#4b2e83'
        ax2.set_ylabel(r'significance $ \frac{S}{\sqrt{B}}$', color=color)
        ax2.plot(x, sig, label='significance', color=color, alpha=1.0)
        ax2.tick_params(axis='y', labelcolor=color)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax1.set_xscale("linear")
        ax2.set_xscale("linear")
        fig.tight_layout()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.3, 1))

        plt.title(f"{label} channel, total significance = {sig_tot:.4f}")


        plt.savefig(f"images_category_id/hh_output_node_histogram_{label}_log_sig_logit.png", dpi=300, bbox_inches='tight')
        plt.show()

        tt_sl.reset()
        tt_dl.reset()
        tt_fh.reset()
        dy.reset()
        hh.reset()

