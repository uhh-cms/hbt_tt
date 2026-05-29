import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the jets emerging from the gen b quarks (similar to the original script)
and creates histograms for the six different background categories emerging for the bb channel:
- for the di-leptonic case, 1, 2 or no fakes are possible.
- for the semileptonic case, 1 or no fakes are possible.
- for the fully hadronic case, only no fakes are possible.
The matching criterion is a delR cut of 0.4 between the gen b quark and the reconstructed b jet.
"""
n_bins = 100

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
events_tt_train = events_tt[events_tt.run3_dnn_moe_hh > 0]#[:100000]

def deltaR(eta1, phi1, eta2, phi2):
    delta_eta = eta2 - eta1
    delta_phi = phi2 - phi1
    # Ensure delta_phi is in the range [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

delr_cut = 0.4 # matched only if distance is smaller than delr = 0.05

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

# define six cases for the tt background
# 1: both b jets matched to gen top b quarks
# corresponds to 2 fakes for the tau in the bb channel
two_matched = events_tt_train[ak.where(ak.count(delta_rs , axis = 1) == 2)]
two_matched_dl = two_matched[ak.where(two_matched.process_id == 1200)]
two_matched_sl = two_matched[ak.where(two_matched.process_id == 1100)]
two_matched_fh = two_matched[ak.where(two_matched.process_id == 1300)]
# 2: only one b jet matched to gen top b quark
one_matched = events_tt_train[ak.where(ak.count(delta_rs, axis = 1) == 1)]
one_matched_dl = one_matched[ak.where(one_matched.process_id == 1200)]
one_matched_sl = one_matched[ak.where(one_matched.process_id == 1100)]
# 3: no b jet matched to gen top b quark
no_matched = events_tt_train[ak.where(ak.count(delta_rs, axis = 1) == 0)]
no_matched_dl = no_matched[ak.where(no_matched.process_id == 1200)]
# some of the events dont belong to any of these cases
other_events = ak.concatenate([one_matched[ak.where(one_matched.process_id == 1300)],
                                no_matched[ak.where(no_matched.process_id == 1100)],
                                no_matched[ak.where(no_matched.process_id == 1300)]])

# plot the btag output score hists
eps = 0#1e-6 # set eps=0 for normal scale
lower_border =0# -14# set to 0 for lin scale
upper_border = 1#12# set to 1 for lin scale
def logit(x):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, lower_border, upper_border-eps)
def inverse_logit(y):
    x = 1 / (1 + np.exp(-y))
    return np.clip(x, eps, 1 - eps)
def identity(x):
    return x
func = identity

# inititalize hists
two_matched_hist_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="2_matched_dl", label="2_matched, dl"))
two_matched_hist_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="2_matched_sl", label="2_matched, sl"))
two_matched_hist_fh = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="2_matched_fh", label="2_matched, fh"))
one_matched_hist_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="1_matched_dl", label="1_matched, dl"))
one_matched_hist_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="1_matched_sl", label="1_matched, sl"))
no_matched_hist_dl  = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="no_matched_dl", label="no_matched, dl"))
all_tt_hist         = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="all", label="all"))
other_ev_hist       = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="other_events", label="other events"))

# fill hists
two_matched_hist_dl.fill(func(two_matched_dl.run3_dnn_moe_hh), weight =two_matched_dl.event_weight)
two_matched_hist_sl.fill(func(two_matched_sl.run3_dnn_moe_hh), weight =two_matched_sl.event_weight)
two_matched_hist_fh.fill(func(two_matched_fh.run3_dnn_moe_hh), weight =two_matched_fh.event_weight)
one_matched_hist_dl.fill(func(one_matched_dl.run3_dnn_moe_hh), weight =one_matched_dl.event_weight)
one_matched_hist_sl.fill(func(one_matched_sl.run3_dnn_moe_hh), weight =one_matched_sl.event_weight)
no_matched_hist_dl.fill(func(no_matched_dl.run3_dnn_moe_hh), weight =no_matched_dl.event_weight)
all_tt_hist.fill(func(events_tt_train.run3_dnn_moe_hh), weight =events_tt_train.event_weight)
other_ev_hist.fill(func(other_events.run3_dnn_moe_hh), weight =other_events.event_weight)
# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)

color = 'black'
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color=color)
# ax1.bar(x, no_matched_hist.values(), width=(upper_border - lower_border) / n_bins, bottom=bottom, alpha=0, label='tt, no matched b jets', color='violet', edgecolor='black')
ax1.step(x, list(two_matched_hist_dl.values()), label='b: 2 fakes, dl', color='tab:blue')
ax1.step(x, list(two_matched_hist_sl.values()), label='b: 2 fakes, sl', color='tab:orange')
ax1.step(x, list(two_matched_hist_fh.values()), label='b: 2 fakes, fh', color='tab:green')
ax1.step(x, list(one_matched_hist_dl.values()), label='b: 1 fakes, dl', color='tab:pink')
ax1.step(x, list(one_matched_hist_sl.values()), label='b: 1 fakes, sl', color='tab:purple')
ax1.step(x, list(no_matched_hist_dl.values()), label='b: 0 fakes, dl', color='tab:brown')
ax1.step(x, list(other_ev_hist.values()), label='b: other events', color='tab:olive')

ax1.step(x, list(all_tt_hist.values()), label='all events', color='red')
ax1.fill_between(x, list(all_tt_hist.values()), color='red', alpha=0.1)



ax1.tick_params(axis='y', labelcolor=color)
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(f"HH output node; tt background events split in nb of b fakes and process (Gen matching criterion $\Delta R < {delr_cut}$ for fakes)")
plt.savefig("6_cases/tt_6_cases_jets_lin", dpi=300, bbox_inches='tight')
plt.show()
