import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script analyses the mutau channel and matches mu, tau to gen W children.
We focus on the semi-leptonic and di-leptonic channels, as mutau is only possible there.
For the different cases, we identify if the muons and taus are fakes or not, with the aim of
analysing the different cases concerning their HH DNN output score distribution.
"""
n_bins = 100

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

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/tt_22pre_v14.parquet")
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_tt = events_tt[events_tt.channel_id == 2] # mutau channel
events_tt_train = events_tt[:1000]
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
delr_cut_tau = 0.3 # matched only if distance is smaller than delr_cut
# events_tau_train = events_tt_train[events_tt_train.gen_top_w_children_pdgId == 13] # only consider muon events
delr_cut_mu = 0.25
# gen matching for muons
delr1_emu = deltaR(
    events_tt_train.emu_eta[:],
    events_tt_train.emu_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 0, 0],
    events_tt_train.gen_top_w_children_phi[:, 0, 0],
)
delr2_emu = deltaR(
    events_tt_train.emu_eta[:],
    events_tt_train.emu_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 0, 1],
    events_tt_train.gen_top_w_children_phi[:, 0, 1],
)
delr3_emu = deltaR(
    events_tt_train.emu_eta[:],
    events_tt_train.emu_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 1, 0],
    events_tt_train.gen_top_w_children_phi[:, 1, 0],
)
delr4_emu = deltaR(
    events_tt_train.emu_eta[:],
    events_tt_train.emu_phi[:],
    events_tt_train.gen_top_w_children_eta[:, 1, 1],
    events_tt_train.gen_top_w_children_phi[:, 1, 1],
)

min_delr_emu1 = np.minimum(delr1_emu, delr2_emu) # first W
min_delr_emu2 = np.minimum(delr3_emu, delr4_emu) # second W
# merge to one array for min delta r of both bjets
delta_rs = np.stack([min_delr_emu1, min_delr_emu2], axis=1)
delta_rs = ak.Array(delta_rs)


##################################################################################################################
# find good cut value for muon matching by looking at the delR distribution
all_delrs = ak.concatenate([ak.flatten(delr1_emu),
                            ak.flatten(delr2_emu),
                            ak.flatten(delr3_emu),
                            ak.flatten(delr4_emu)])

alldelr_max = 4.5
alldelr_nbins = 100

alldelr = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R"))
alldelr.fill(all_delrs)

x = np.linspace(0, alldelr_max, alldelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
fig = plt.figure(figsize=(10, 6))
plt.bar(x, alldelr.values(), width=(alldelr_max)/alldelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
plt.xticks(xticks)
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of all reconstructed muons with gen W children")

plt.savefig(f"analysis_mutau/alldelrs_mu_distribution", dpi=300, bbox_inches='tight')
plt.show()
alldelr.reset()


##################################################################################################################
# array with all delrs, to not loose the indices
delta_rs1 = np.column_stack([delr1_emu, delr2_emu])
delta_rs2 = np.column_stack([delr3_emu, delr4_emu])
delta_rs = np.stack([delta_rs1, delta_rs2], axis=1)
delta_rs = ak.Array(delta_rs)
from IPython import embed; embed()

mask_genmatched_mus = (delta_rs < delr_cut_mu)
mask_mu = ((events_tt_train.gen_top_w_children_pdgId == 13) 
         | (events_tt_train.gen_top_w_children_pdgId == -13))

# TODO: shape mismatch between mask_genmatched_mus and events_tt_train, as mask_genmatched_mus has shape (n_events, 2, 2) and events_tt_train has shape (n_events,). 
# We need to reduce the mask_genmatched_mus to shape (n_events,) by checking if any of the 4 delR values is smaller than the cut value. 
# This can be done by using ak.any(mask_genmatched_mus, axis=(1, 2)) to get a boolean array of shape (n_events,) that indicates if there is at least one gen-matched muon in
matched_mus         = events_tt_train[ak.any(mask_genmatched_mus, axis=(1, 2)) & mask_mu]
matched_mu_fakes    = events_tt_train[ak.any(mask_genmatched_mus, axis=(1, 2)) & ~mask_mu]
unmatched           = events_tt_train[~ak.any(mask_genmatched_mus, axis=(1, 2))]

# delta_rs = delta_rs[mask_genmatched_mus]
# pdg_ids = events_tt_train.gen_top_w_children_pdgId[mask_genmatched_mus]

# mask_mu_event = ak.any(mask_mu, axis=-1)
# mask_no_mu = ~mask_mu_event

# mu_indices = ak.where(mask_mu_event)[0]
# events_mu = events_tt_train[mu_indices]
# mu_fake_indices = ak.where(mask_no_mu)[0]
# events_mu_fakes = events_tt[mu_fake_indices]

# initialize hists
mu_events = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="mu", label="mu"))
mu_fakes  = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="mu_fakes", label="mu_fakes"))
unmatched = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="unmatched", label="unmatched"))
all_events = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="all_events", label="all_events"))

# fill hists
mu_events.fill(func(matched_mus.run3_dnn_moe_hh), weight =matched_mus.event_weight)
mu_fakes.fill(func(matched_mu_fakes.run3_dnn_moe_hh), weight =matched_mu_fakes.event_weight)
unmatched.fill(func(unmatched.run3_dnn_moe_hh), weight =unmatched.event_weight)
all_events.fill(func(events_tt.run3_dnn_moe_hh), weight =events_tt.event_weight)

# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.bar(x, mu_events.values(), width=(upper_border - lower_border) / n_bins, bottom=None, alpha=0.5, label='muon events in mutau channel', color='green', edgecolor='black')
ax1.bar(x, mu_fakes.values(), width=(upper_border - lower_border) / n_bins, bottom=None, alpha=0.5, label='muon fake events in mutau channel', color='orange', edgecolor='black')
ax1.bar(x, unmatched.values(), width=(upper_border - lower_border) / n_bins, bottom=None, alpha=0.5, label='unmatched events', color='blue', edgecolor='black')
# ax1.bar(x, all_matched.values(), width=(upper_border - lower_border) / n_bins, bottom=None, alpha=0.5, label='all events in mutau channel', color='red', edgecolor='red')
ax1.step(x, list(all_events.values()), label='all events', color='red')
ax1.fill_between(x, list(all_events.values()), color='red', alpha=0.1)

ax1.tick_params(axis='y', labelcolor="black")
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(f"HH output node; tt background events split in correctly identified tau events and tau fake events$")
plt.savefig("analysis_mutau/tau_fakes", dpi=300, bbox_inches='tight')
plt.show()
# from IPython import embed; embed()
# mu_matched_events = events_tt_train[mu_mask]
# mask = delta_rs < delr_cut_mu
# delta_rs_mu = delta_rs[mask]

# initialize histograms
