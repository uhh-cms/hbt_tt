import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the leptons emerging from the W decay.
"""
n_bins = 50

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/tt_22pre_v14.parquet")
events_tt_train = events_tt[events_tt.run3_dnn_moe_hh > 0][:100]

# important columns
# events_tt_train.gen_top_w_children_eta
# events_tt_train.gen_top_w_children_phi
# events_tt_train.emu_eta
# events_tt_train.emu_phi
# events_tt_train.tau_eta
# events_tt_train.tau_phi
# events_tt_train.tau_genPartFlav
print(events_tt_train.tau_eta)
from IPython import embed; embed(header="MESSAGE Line 23 | File: genmatching_2leptons.py")
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
########################################################
# plot delrs for ALL taus to find a good delr cut value
all_delrs = ak.concatenate([ak.flatten(delr1_tau),
                            ak.flatten(delr2_tau),
                            ak.flatten(delr3_tau),
                            ak.flatten(delr4_tau)])

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
plt.title("Delta R of all reconstructed taus with gen W children")

plt.savefig(f"images_leptons/alldelrs_tau_distribution", dpi=300, bbox_inches='tight')
plt.show()
alldelr.reset()
##########################################################################################################
# apply delr cut to define matched taus
# match emu to smallest distance gen top W children
# min_delr_emu1 = np.minimum(delr1_emu, delr2_emu)
# min_delr_emu2 = np.minimum(delr3_emu, delr4_emu)
# min_delr_emu = np.minimum(min_delr_emu1, min_delr_emu2)
# delta_rs_emu = ak.Array(min_delr_emu)

# match tau to smallest distance gen top W children
min_delr_tau1 = np.minimum(delr1_tau, delr2_tau)
min_delr_tau2 = np.minimum(delr3_tau, delr4_tau)
min_delr_tau = np.minimum(min_delr_tau1, min_delr_tau2) # matched tau events with only one entry
delta_rs_tau = ak.Array(min_delr_tau)

# matched taus have a delta r smaller than delr_cut to a gen W children
mask = delta_rs_tau < delr_cut
delta_rs_tau = delta_rs_tau[mask]

# plot delr distribution
delr_hist = ak.flatten(delta_rs_tau, axis=None)
delr = Hist(hist.axis.Regular(n_bins, 0, delr_cut, name="", label="delta R"))
delr.fill(delr_hist)

x = np.linspace(0, delr_cut, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig = plt.figure(figsize=(10, 6))
plt.bar(x, delr.values(), width=(delr_cut)/n_bins, bottom=None, fill=True,  color='pink', edgecolor='black')

plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of reconstructed taus with gen W children")

plt.savefig(f"images_leptons/delr_tau_distribution", dpi=300, bbox_inches='tight')
plt.show()
delr.reset()

##################################################################################################################
# define three classes of tt bg
# 1: both b jets matched to gen top b quarks
two_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(delta_rs_tau , axis = 1) == 2)]
# 2: only one b jet matched to gen top b quark
one_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(delta_rs_tau, axis = 1) == 1)]
# 3: no b jet matched to gen top b quark
no_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(delta_rs_tau, axis = 1) == 0)]

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
two_matched_hist.fill(func(two_matched), weight =events_tt_train.event_weight[ak.where(ak.count(delta_rs_tau, axis = 1) == 2)])
one_matched_hist.fill(func(one_matched), weight =events_tt_train.event_weight[ak.where(ak.count(delta_rs_tau, axis = 1) == 1)])
no_matched_hist.fill(func(no_matched), weight =events_tt_train.event_weight[ak.where(ak.count(delta_rs_tau, axis = 1) == 0)])

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
plt.title(f"HH output node; tt background events split in nb of matched taus, matching criterion: $\Delta R < {delr_cut}$")
plt.savefig("images_leptons/tt_genmatched_taus", dpi=300, bbox_inches='tight')
plt.show()

# define three event classes and weights
events_two_matched = events_tt_train[ak.where(ak.count(delta_rs_tau, axis = 1) == 2)]
events_one_matched = events_tt_train[ak.where(ak.count(delta_rs_tau, axis = 1) == 1)]
events_no_matched = events_tt_train[ak.where(ak.count(delta_rs_tau, axis = 1) == 0)]
weight_2matched = events_tt_train.event_weight[ak.where(ak.count(delta_rs_tau, axis = 1) == 2)]
weight_1matched = events_tt_train.event_weight[ak.where(ak.count(delta_rs_tau, axis = 1) == 1)]
weight_0matched = events_tt_train.event_weight[ak.where(ak.count(delta_rs_tau, axis = 1) == 0)]

tautau_mask_res1b_tt2 = ak.any(events_two_matched.category_ids == 203, axis = 1)
tautau_mask_res1b_tt1 = ak.any(events_one_matched.category_ids == 203, axis = 1)
tautau_mask_res1b_tt0 = ak.any(events_no_matched.category_ids == 203, axis = 1)
tautau_mask_res1b_dy = ak.any(events_dy.category_ids == 203, axis = 1)
tautau_mask_res1b_hh = ak.any(events_hh.category_ids == 203, axis = 1)

tautau_mask_res2b_tt2 = ak.any(events_two_matched.category_ids == 207, axis = 1)
tautau_mask_res2b_tt1 = ak.any(events_one_matched.category_ids == 207, axis = 1)
tautau_mask_res2b_tt0 = ak.any(events_no_matched.category_ids == 207, axis = 1)
tautau_mask_res2b_dy = ak.any(events_dy.category_ids == 207, axis = 1)
tautau_mask_res2b_hh = ak.any(events_hh.category_ids == 207, axis = 1)

masks = [[tautau_mask_res1b_tt2, tautau_mask_res1b_tt1, tautau_mask_res1b_tt0, tautau_mask_res1b_dy, tautau_mask_res1b_hh],
         [tautau_mask_res2b_tt2, tautau_mask_res2b_tt1, tautau_mask_res2b_tt0, tautau_mask_res2b_dy, tautau_mask_res2b_hh]]
labels = ["tautau, res 1b", "tautau, res 2b"]

for mask, label in zip(masks, labels):
    upper_border = 11
    lower_border = -10
    x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    # initialize histograms
    hh =               Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="hh", label="hh"))
    dy =               Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="dy", label="dy"))
    two_matched_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="2_matched", label="2_matched"))
    one_matched_hist = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="1_matched", label="1_matched"))
    no_matched_hist =  Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="no_matched", label="no_matched"))

    # fill hists
    two_matched_hist.fill(func(two_matched[mask[0]]), weight=weight_2matched[mask[0]])
    one_matched_hist.fill(func(one_matched[mask[1]]), weight=weight_1matched[mask[1]])
    no_matched_hist.fill(func(no_matched[mask[2]]), weight=weight_0matched[mask[2]])
    dy.fill(func(events_dy.run3_dnn_moe_hh[mask[3]]), weight =events_dy.event_weight[mask[3]])
    hh.fill(func(events_hh.run3_dnn_moe_hh[mask[4]]), weight =events_hh.event_weight[mask[4]])
    # scale the hh histogram up, weighted by the integral of the dy and tt data
    scaling_factor = (hh.values().sum() / (two_matched_hist.values().sum() + one_matched_hist.values().sum() + no_matched_hist.values().sum() + dy.values().sum()))**(-1)
    # plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(right=0.85)
    color = 'black'
    bottom = np.zeros_like(x)
    ax1.bar(x, two_matched_hist.values(), width=(22)/n_bins, bottom=bottom, alpha=0.5, color="deeppink", label='tt, two matched b jets',  edgecolor='black')
    bottom+=two_matched_hist.values()
    ax1.bar(x, one_matched_hist.values(), width=(22)/n_bins, bottom=bottom, alpha=0.5, color='mediumblue', label='tt, one matched b jet',  edgecolor='black')
    bottom+=one_matched_hist.values()
    ax1.bar(x, no_matched_hist.values(), width=(22)/n_bins, bottom=bottom, alpha=0.5, color='violet', label='tt, no matched b jets',  edgecolor='black')
    bottom+=no_matched_hist.values()
    ax1.bar(x, dy.values(), width=(22)/n_bins, bottom=bottom, alpha=0.5, color='green', label='dy', edgecolor='black')
    ax1.bar(x, hh.values() * scaling_factor, width=(22)/n_bins, bottom=None, fill=False, label=f'hh x ({scaling_factor:.2f})', color='green', edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel("logit of HH output node")
    ax1.set_ylabel("Number of events")
    ax1.set_ylim(bottom=1e-1)
    ax1.set_xlim(left=lower_border)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.set_yscale("log")
    fig.tight_layout()
    plt.legend()
    plt.title(f"{label} channel; tt background events split in nb of matched taus, matching criterion: $\Delta R < {delr_cut}$")
    plt.savefig(f"images_leptons/tt_genmatched_taus_{label}", dpi=300, bbox_inches='tight')
    plt.show()

    two_matched_hist.reset()
    one_matched_hist.reset()
    no_matched_hist.reset()
    dy.reset()
    hh.reset()
#############################################################################################################################
################################################################################################################################
# do the same with events_tt_train.tau_genPartFlav on x axis instead of NN output score
# plot bar chart for origin of gen matched taus
mask = delta_rs_tau < delr_cut
for hh_cut, label in zip([0, 0.5, 0.7, 0.9], ["none", "loose", "tight", "very tight"]):
    matched_events = events_tt_train[ak.flatten(mask)]
    events_tt_cut = matched_events[matched_events.run3_dnn_moe_hh > hh_cut]

    matched_weights = events_tt_train.event_weight[ak.flatten(mask)]
    matched_weights_cut = matched_weights[events_tt_cut.run3_dnn_moe_hh > hh_cut]

    # weights in the correct shape for np.bincount
    weights_2d = ak.zeros_like(events_tt_cut.tau_genPartFlav)+matched_weights_cut
    binned_matched = np.bincount(ak.to_numpy(ak.flatten(events_tt_cut.tau_genPartFlav)), weights=ak.flatten(weights_2d))

    # plot
    x = np.arange(6)
    labels = ["Unknown",
            "Prompt e",
            r"Prompt $\mu$",
            r"$\tau_e$",
            r"$\tau_\mu$",
            r"$\tau_h$"]
    plt.figure(figsize=(10, 6))
    plt.bar(x, binned_matched, color='blueviolet', alpha=0.5, edgecolor='black')
    plt.xlabel("origin")
    plt.ylabel("Number of events")
    plt.title(f"Origin of gen matched taus; HH output node wp: {label} ({hh_cut})")
    plt.xticks(x, labels, rotation=45)
    plt.yscale('linear')
    plt.savefig(f"images_leptons/origin_genmatched_taus_{label}", dpi=300, bbox_inches='tight')
    plt.show()





