import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the b quarks emerging from the tt background.
"""
n_bins = 50

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
events_tt_train = events_tt[events_tt.run3_dnn_moe_hh > 0]#[:100000]

# important columns
# events_tt_train.bjet_eta
# events_tt_train.bjet_phi
# events_tt_train.bjet_btag
# events_tt.gen_top_b_eta
# events_tt.gen_top_b_phi

def deltaR(eta1, phi1, eta2, phi2):
    delta_eta = eta2 - eta1
    delta_phi = phi2 - phi1
    # Ensure delta_phi is in the range [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

delr_cut = 0.05 # matched only if distance is smaller than delr = 0.05

delta_r1 = deltaR(
    events_tt_train.bjet_eta[:, 0],
    events_tt_train.bjet_phi[:, 0],
    events_tt_train.gen_top_b_eta[:, 0],
    events_tt_train.gen_top_b_phi[:, 0],
)

delta_r2 = deltaR(
    events_tt_train.bjet_eta[:, 0],
    events_tt_train.bjet_phi[:, 0],
    events_tt_train.gen_top_b_eta[:, 1],
    events_tt_train.gen_top_b_phi[:, 1],
)

delta_r3 = deltaR(
    events_tt_train.bjet_eta[:, 1],
    events_tt_train.bjet_phi[:, 1],
    events_tt_train.gen_top_b_eta[:, 0],
    events_tt_train.gen_top_b_phi[:, 0],
)

delta_r4 = deltaR(
    events_tt_train.bjet_eta[:, 1],
    events_tt_train.bjet_phi[:, 1],
    events_tt_train.gen_top_b_eta[:, 1],
    events_tt_train.gen_top_b_phi[:, 1],
)
# match bjet to smallest distance gen b quark
min_delr_bjet1 = np.minimum(delta_r1, delta_r2)
min_delr_bjet2 = np.minimum(delta_r3, delta_r4)
# merge to one array for min delta r of both bjets
delta_rs = np.stack([min_delr_bjet1, min_delr_bjet2], axis=1)
delta_rs = ak.Array(delta_rs)

# matched b jets have a delta r smaller than delr_cut to a gen top b quark
mask = delta_rs < delr_cut
delta_rs = delta_rs[mask]
btags_of_matched_events = events_tt_train.bjet_btag[mask]
# ev_tt_obj_indices = ak.local_index(events_tt_train.bjet_eta)[mask]

# hist to plot delta r distribution of matched b jets
delr_hist = ak.flatten(delta_rs, axis=None)
delr = Hist(hist.axis.Regular(n_bins, 0, delr_cut, name="", label="delta_r"))
delr.fill(delr_hist)

x = np.linspace(0, delr_cut, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig = plt.figure(figsize=(10, 6))
plt.bar(x, delr.values(), width=(delr_cut)/n_bins, bottom=None, fill=True,  color='pink', edgecolor='black')#, label=f'hh x ({scaling_factor:.2f})')

plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of both gen b jets with matched gen top b quark")

plt.savefig(f"images/delr_2jets_hist", dpi=300, bbox_inches='tight')
plt.show()
delr.reset()

# define three event classes
# 1: both b jets matched to gen top b quarks
two_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(btags_of_matched_events, axis = 1) == 2)]
# 2: only one b jet matched to gen top b quark
one_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(btags_of_matched_events, axis = 1) == 1)]
# 3: no b jet matched to gen top b quark
no_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(btags_of_matched_events, axis = 1) == 0)]

# plot the btag output score hists
eps = 1e-6 # set eps=0 for normal scale
lower_border = -14# set to 0 for lin scale
upper_border = 9# set to 1 for lin scale
def logit(x):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, lower_border, upper_border-eps)
def identity(x):
    return x
func = logit

# inititalize hists
events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data
events_hh = events_hh[events_hh.run3_dnn_moe_hh > 0]
events_dy = events_dy[events_dy.run3_dnn_moe_hh > 0]

hh =               Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="hh", label="hh"))
dy =               Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="dy", label="dy"))
two_matched_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="2_matched", label="2_matched"))
one_matched_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="1_matched", label="1_matched"))
no_matched_hist =  Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="no_matched", label="no_matched"))

# fill hists
hh.fill(func(events_hh.run3_dnn_moe_hh), weight =events_hh.event_weight)
dy.fill(func(events_dy.run3_dnn_moe_hh), weight =events_dy.event_weight)
two_matched_hist.fill(func(two_matched), weight =events_tt_train.event_weight[ak.where(ak.count(btags_of_matched_events, axis = 1) == 2)])
one_matched_hist.fill(func(one_matched), weight =events_tt_train.event_weight[ak.where(ak.count(btags_of_matched_events, axis = 1) == 1)])
no_matched_hist.fill(func(no_matched), weight =events_tt_train.event_weight[ak.where(ak.count(btags_of_matched_events, axis = 1) == 0)])

# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

# scale the hh histogram up, weighted by the integral of the dy and tt data
scaling_factor = (hh.values().sum() / (two_matched_hist.values().sum() + one_matched_hist.values().sum() + no_matched_hist.values().sum() + dy.values().sum()))**(-1)

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)

color = 'black'
bottom = np.zeros_like(x)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color=color)
ax1.bar(x, no_matched_hist.values(), width=(upper_border - lower_border) / n_bins, bottom=bottom, alpha=0.5, label='tt, no matched b jets', color='violet', edgecolor='black')
bottom += no_matched_hist.values()
ax1.bar(x, one_matched_hist.values(), width=(upper_border - lower_border) / n_bins, bottom=bottom, alpha=0.5, label='tt, one matched b jet', color='mediumblue', edgecolor='black')
bottom += one_matched_hist.values()
ax1.bar(x, two_matched_hist.values(), width=(upper_border - lower_border) / n_bins, bottom=bottom, alpha=0.5, label='tt, two matched b jets', color='deeppink', edgecolor='black')
bottom += two_matched_hist.values()
ax1.bar(x, dy.values(), width=(upper_border - lower_border) / n_bins, bottom=bottom, alpha=0.5, label='dy', color='green', edgecolor='black')
ax1.bar(x, hh.values() * scaling_factor, width=(upper_border - lower_border) / n_bins, bottom=None, fill=False, label=f'hh x ({scaling_factor:.2f})', color='red', edgecolor='black')

ax1.tick_params(axis='y', labelcolor=color)
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title("HH output node; tt background events split in nb of gen matched b jets")
plt.savefig("images/tt_genmatched_split_hist_logit", dpi=300, bbox_inches='tight')
plt.show()
