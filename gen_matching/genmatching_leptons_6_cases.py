import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the leptons emerging from the W decay (similar to the original script)
and creates histograms for the six different background categories emerging for the tautau channel:
- for the di-leptonic case, 1, 2 or no fakes are possible.
- for the semileptonic case, 1 or no fakes are possible.
- for the fully hadronic case, only no fakes are possible.
The matching criterion is a delR cut of 0.3 between the gen tau and the reconstructed tau.
"""
n_bins = 100

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/tt_22pre_v14.parquet")
events_tt_train = events_tt[events_tt.run3_dnn_moe_hh > 0]

# important columns
# events_tt_train.gen_top_w_children_eta
# events_tt_train.gen_top_w_children_phi
# events_tt_train.emu_eta
# events_tt_train.emu_phi
# events_tt_train.tau_eta
# events_tt_train.tau_phi
# events_tt_train.tau_genPartFlav

def deltaR(eta1, phi1, eta2, phi2):
    delta_eta = eta2 - eta1
    delta_phi = phi2 - phi1
    # Ensure delta_phi is in the range [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

delr_cut = 0.3 # matched only if distance is smaller than delr_cut
# emu gen matching
# delr1_emu = deltaR(
#     events_tt_train.emu_eta[:],
#     events_tt_train.emu_phi[:],
#     events_tt_train.gen_top_w_children_eta[:, 0, 0],
#     events_tt_train.gen_top_w_children_phi[:, 0, 0],
# )
# delr2_emu = deltaR(
#     events_tt_train.emu_eta[:],
#     events_tt_train.emu_phi[:],
#     events_tt_train.gen_top_w_children_eta[:, 0, 1],
#     events_tt_train.gen_top_w_children_phi[:, 0, 1],
# )
# delr3_emu = deltaR(
#     events_tt_train.emu_eta[:],
#     events_tt_train.emu_phi[:],
#     events_tt_train.gen_top_w_children_eta[:, 1, 0],
#     events_tt_train.gen_top_w_children_phi[:, 1, 0],
# )
# delr4_emu = deltaR(
#     events_tt_train.emu_eta[:],
#     events_tt_train.emu_phi[:],
#     events_tt_train.gen_top_w_children_eta[:, 1, 1],
#     events_tt_train.gen_top_w_children_phi[:, 1, 1],
# )
# tau gen matching
delr1_tau = deltaR(
    events_tt_train.tau_eta[:],
    events_tt_train.tau_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 0, 0],
    events_tt_train.gen_top_w_children_phi[:, 0, 0],
)
delr2_tau = deltaR(
    events_tt_train.tau_eta[:],
    events_tt_train.tau_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 0, 1],
    events_tt_train.gen_top_w_children_phi[:, 0, 1],
)
delr3_tau = deltaR(
    events_tt_train.tau_eta[:],
    events_tt_train.tau_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 1, 0],
    events_tt_train.gen_top_w_children_phi[:, 1, 0],
)
delr4_tau = deltaR(
    events_tt_train.tau_eta[:],
    events_tt_train.tau_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 1, 1],
    events_tt_train.gen_top_w_children_phi[:, 1, 1],
)
all_delrs = ak.concatenate([ak.flatten(delr1_tau),
                            ak.flatten(delr2_tau),
                            ak.flatten(delr3_tau),
                            ak.flatten(delr4_tau)])

# match tau to smallest distance gen top W children
min_delr_tau1 = np.minimum(delr1_tau, delr2_tau)
min_delr_tau2 = np.minimum(delr3_tau, delr4_tau)
min_delr_tau = np.minimum(min_delr_tau1, min_delr_tau2) # matched tau events with only one entry
delta_rs_tau = ak.Array(min_delr_tau)

# matched taus have a delta r smaller than delr_cut to a gen W children
mask = delta_rs_tau < delr_cut
delta_rs_tau = delta_rs_tau[mask]

##################################################################################################################
# define six classes of tt bg
# 1: both b jets matched to gen top b quarks
# corresponds to 2 fakes
# two_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(delta_rs_tau , axis = 1) == 2)]
two_matched = events_tt_train[ak.where(ak.count(delta_rs_tau , axis = 1) == 2)]
two_matched_dl = two_matched[ak.where(two_matched.process_id == 1200)]
two_matched_sl = two_matched[ak.where(two_matched.process_id == 1100)]
two_matched_fh = two_matched[ak.where(two_matched.process_id == 1300)]
# 2: only one b jet matched to gen top b quark
# corresponds to 1 fake
# one_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(delta_rs_tau, axis = 1) == 1)]
one_matched = events_tt_train[ak.where(ak.count(delta_rs_tau, axis = 1) == 1)]
one_matched_dl = one_matched[ak.where(one_matched.process_id == 1200)]
one_matched_sl = one_matched[ak.where(one_matched.process_id == 1100)]
# 3: no b jet matched to gen top b quark
# corresponds to 0 fakes
# no_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(delta_rs_tau, axis = 1) == 0)]
no_matched = events_tt_train[ak.where(ak.count(delta_rs_tau, axis = 1) == 0)]
no_matched_dl = no_matched[ak.where(no_matched.process_id == 1200)]
# some of the events dont belong to any of these cases
other_events = ak.concatenate([one_matched[ak.where(one_matched.process_id == 1300)],
                                no_matched[ak.where(no_matched.process_id == 1100)],
                                no_matched[ak.where(no_matched.process_id == 1300)]])

# plot the signal output score hists with the 3 classes
eps = 1e-6 # set eps=0 for normal scale
lower_border = -14# set to 0 for lin scale
upper_border = 12# set to 1 for lin scale
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

two_matched_hist_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="2_matched_dl", label="2_matched, dl"))
# inititalize hists
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
ax1.step(x, list(two_matched_hist_dl.values()), label=r'$\tau$: 2 fakes, dl', color='tab:blue')
ax1.step(x, list(two_matched_hist_sl.values()), label=r'$\tau$: 2 fakes, sl', color='tab:orange')
ax1.step(x, list(two_matched_hist_fh.values()), label=r'$\tau$: 2 fakes, fh', color='tab:green')
ax1.step(x, list(one_matched_hist_dl.values()), label=r'$\tau$: 1 fakes, dl', color='tab:pink')
ax1.step(x, list(one_matched_hist_sl.values()), label=r'$\tau$: 1 fakes, sl', color='tab:purple')
ax1.step(x, list(no_matched_hist_dl.values()), label=r'$\tau$: 0 fakes, dl', color='tab:brown')
ax1.step(x, list(other_ev_hist.values()), label=r'$\tau$: other events', color='tab:olive')

ax1.step(x, list(all_tt_hist.values()), label='all events', color='red')
ax1.fill_between(x, list(all_tt_hist.values()), color='red', alpha=0.1)

ax1.tick_params(axis='y', labelcolor=color)
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(f"HH output node; tt background events split in nb of tau fakes and process (Gen matching criterion $\Delta R < {delr_cut}$ for fakes)")
plt.savefig("6_cases/tt_6_cases_leptons", dpi=300, bbox_inches='tight')
plt.show()

##############################################################################################################
# plot MET for different cases
# inititalize hists
max_x_value = 400#max(ak.to_numpy(two_matched_dl.met_pt))
met_hist = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="met", label="met"))
met_hist_2_dl  = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="2 fakes, dl", label="2 fakes, dl"))
met_hist_2_sl = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="2 fakes, sl", label="2 fakes, sl"))
met_hist_2_fh = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="2 fakes, fh", label="2 fakes, fh"))
met_hist_1_dl = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="1 fake, dl", label="1 fake, dl"))
met_hist_1_sl = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="1 fake, sl", label="1 fake, sl"))
met_hist_0_dl = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="0 fakes, dl", label="0 fakes, dl"))
met_hist_others = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="others", label="others"))
met_hist_2_matched = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="2_matched", label="2_matched"))
met_hist_1_matched = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="1_matched", label="1_matched"))
met_hist_0_matched = Hist(hist.axis.Regular(n_bins, 0, max_x_value, name="0_matched", label="0_matched"))

# fill hists
met_hist.fill(events_tt_train.met_pt, weight =events_tt_train.event_weight)
met_hist_2_dl.fill(two_matched_dl.met_pt, weight = two_matched_dl.event_weight)
met_hist_2_sl.fill(two_matched_sl.met_pt, weight = two_matched_sl.event_weight)
met_hist_2_fh.fill(two_matched_fh.met_pt, weight = two_matched_fh.event_weight)
met_hist_1_dl.fill(one_matched_dl.met_pt, weight = one_matched_dl.event_weight)
met_hist_1_sl.fill(one_matched_sl.met_pt, weight = one_matched_sl.event_weight)
met_hist_0_dl.fill(no_matched_dl.met_pt, weight = no_matched_dl.event_weight)
met_hist_others.fill(other_events.met_pt, weight = other_events.event_weight)
met_hist_2_matched.fill(two_matched.met_pt, weight = two_matched.event_weight)
met_hist_1_matched.fill(one_matched.met_pt, weight = one_matched.event_weight)
met_hist_0_matched.fill(no_matched.met_pt, weight = no_matched.event_weight)

# plot histograms
x = np.linspace(0, max_x_value, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
# ax1.bar(x, met_hist.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='violet', label="all", edgecolor='violet')
ax1.bar(x, met_hist_2_dl.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:blue', label="2 fakes, dl", edgecolor='tab:blue')
ax1.bar(x, met_hist_2_sl.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:orange', label="2 fakes, sl", edgecolor='tab:orange')
ax1.bar(x, met_hist_2_fh.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:green', label="2 fakes, fh", edgecolor='tab:green')
ax1.bar(x, met_hist_1_dl.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:pink', label="1 fake, dl", edgecolor='tab:pink')
ax1.bar(x, met_hist_1_sl.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:purple', label="1 fake, sl", edgecolor='tab:purple')
ax1.bar(x, met_hist_0_dl.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:brown', label="0 fakes, dl", edgecolor='tab:brown')
ax1.bar(x, met_hist_others.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:olive', label="others", edgecolor='tab:olive')
# ax1.bar(x, met_hist_2_matched.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:cyan', label="2 matched", edgecolor='tab:cyan')
# ax1.bar(x, met_hist_1_matched.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:gray', label="1 matched", edgecolor='tab:gray')
# ax1.bar(x, met_hist_0_matched.values(), width=max_x_value / n_bins, alpha=0.5, fill=False, color='tab:orange', label="0 matched", edgecolor='tab:orange')
ax1.set_xlabel('Missing energy traverse')
ax1.set_ylabel('Number of events')
ax1.get_legend_handles_labels()
plt.legend()
fig.tight_layout()
plt.title(f"MET distribution for the different number of fakes (Gen matching criterion $\Delta R < {delr_cut}$ for fakes)")
plt.savefig("other_columns/met_6_cases", dpi=300, bbox_inches='tight')
plt.show()
