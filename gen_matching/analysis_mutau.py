import itertools

import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("analysis_mutau", exist_ok=True)

"""This script analyses the mutau channel and matches mu, tau to gen W children.
We focus on the semi-leptonic and di-leptonic channels, as mutau is only possible there.
For the different cases, we identify if the muons and taus are fakes or not, with the aim of
analysing the different cases concerning their HH DNN output score distribution.
"""
n_bins = 100

eps = 1e-6 # set eps=0 for normal scale
lower_border = -14# set to 0 for lin scale
upper_border = 7# set to 1 for lin scale
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
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_tt = events_tt[events_tt.channel_id == 2] # mutau channel
events_tt_train = events_tt
# events_tt_train = ak.concatenate([events_tt[:10000], events_tt[844445:854446]]) # first ev are dl, second sl
# events_tt_train = ak.concatenate([events_tt_train, events_tt[844127:844444,]]) # also add fh events

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

# alldelr = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R"))
# alldelr.fill(all_delrs)

# x = np.linspace(0, alldelr_max, alldelr_nbins + 1)  # bin edges
# x = (x[:-1] + x[1:]) / 2  # bin centers
# xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
# fig = plt.figure(figsize=(10, 6))
# plt.bar(x, alldelr.values(), width=(alldelr_max)/alldelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
# plt.xticks(xticks)
# plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
# plt.ylabel("Number of events")
# plt.title("Delta R of all reconstructed muons with gen W children")

# plt.savefig(f"analysis_mutau/alldelrs_mu_distribution", dpi=300, bbox_inches='tight')
# plt.bar(x, alldelr.values(), width=(alldelr_max)/alldelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
# plt.xticks(xticks)
# plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
# plt.ylabel("Number of events")
# plt.show()
# alldelr.reset()


##################################################################################################################
# muon matching
# event passes muon matching if at least one muon is matched
# array with all delrs, to not loose the indices
delta_rs1 = np.column_stack([delr1_emu, delr2_emu])
delta_rs2 = np.column_stack([delr3_emu, delr4_emu])
delta_rs = np.stack([delta_rs1, delta_rs2], axis=1)
delta_rs = ak.Array(delta_rs)

# event-wise loop to find events which definitely hae one genmatched mu
# vectorised way is very difficult because indices get lost very easily
pdgids = events_tt_train.gen_top_w_children_pdgId
matched_mu_indices = []
w_mu_indices = []
for i in range(len(events_tt_train)):
    if ((delta_rs[i][0][0] < delr_cut_mu) and (abs(pdgids[i][0][0])== 13)) or ((delta_rs[i][0][1] < delr_cut_mu) and (abs(pdgids[i][0][1])== 13)):
        matched_mu_indices.append(i)
        w_mu_indices.append(0) # muon from first W
        pass
    elif ((delta_rs[i][1][0] < delr_cut_mu) and (abs(pdgids[i][1][0])== 13)) or ((delta_rs[i][1][1] < delr_cut_mu) and (abs(pdgids[i][1][1])== 13)):
        matched_mu_indices.append(i)
        w_mu_indices.append(1) # muon from second W


unmatched_mu_indices = [i for i in range(len(events_tt_train)) if i not in matched_mu_indices]

matched_mu_events = events_tt_train[matched_mu_indices]
fake_mu_events = events_tt_train[unmatched_mu_indices]

matched_mu_dl = matched_mu_events[matched_mu_events.process_id == 1200]
matched_mu_sl = matched_mu_events[matched_mu_events.process_id == 1100]
matched_mu_fh = matched_mu_events[matched_mu_events.process_id == 1300]
fake_mu_dl = fake_mu_events[fake_mu_events.process_id == 1200]
fake_mu_sl = fake_mu_events[fake_mu_events.process_id == 1100]
fake_mu_fh = fake_mu_events[fake_mu_events.process_id == 1300]

# initialize hists
# fh match hist not necessary, as all fh events are unmatched (which we expected)
mu_events_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="mu", label="mu"))
mu_events_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="mu", label="mu"))
mu_fakes_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="mu_fakes", label="mu_fakes"))
mu_fakes_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="mu_fakes", label="mu_fakes"))
mu_fakes_fh = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="mu_fakes", label="mu_fakes"))
all_events = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="all_events", label="all_events"))
# fill hists
mu_events_dl.fill(func(matched_mu_dl.run3_dnn_moe_hh), weight =matched_mu_dl.event_weight)
mu_events_sl.fill(func(matched_mu_sl.run3_dnn_moe_hh), weight =matched_mu_sl.event_weight)
mu_fakes_dl.fill(func(fake_mu_dl.run3_dnn_moe_hh), weight =fake_mu_dl.event_weight)
mu_fakes_sl.fill(func(fake_mu_sl.run3_dnn_moe_hh), weight =fake_mu_sl.event_weight)
mu_fakes_fh.fill(func(fake_mu_fh.run3_dnn_moe_hh), weight =fake_mu_fh.event_weight)
all_events.fill(func(events_tt.run3_dnn_moe_hh), weight =events_tt.event_weight)

# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, list(mu_events_dl.values()), alpha=0.9, label=r'matched dl mu events', color='green')
ax1.step(x, list(mu_events_sl.values()), alpha=0.9, label=r'matched sl mu events', color='mediumseagreen')
ax1.step(x, list(mu_fakes_dl.values()), alpha=0.9, label=r'fake dl mu events', color='red')
ax1.step(x, list(mu_fakes_sl.values()), alpha=0.9, label=r'fake sl mu events', color='slateblue')
ax1.step(x, list(mu_fakes_fh.values()), alpha=0.9, label=r'fake fh mu events', color='purple')

ax1.step(x, list(all_events.values()), label='all events in mutau channel', color='blue')
# ax1.fill_between(x, list(all_events.values()), color='red', alpha=0.1)

ax1.tick_params(axis='y', labelcolor="black")
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(r"HH output node; mutau channel tt bg split in correctly matched and fake muon events (matching criterion: $\Delta R <$"+f" {delr_cut_mu})")
plt.savefig("analysis_mutau/dnn_mu_matching_allevents", dpi=300, bbox_inches='tight')
plt.show()


###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# semi-leptonic case: tau matching
# for the semi-leptonic case, if the mu is correct, the tau HAS TO BE FAKE bc the other W needs to decay hadronically,
# with the jets emerging from the hadronic decaying W being misidentified as tau_had

# find the indices of the hadronic decaying W, which is the one leading to the fake tau
# w_mu_indices is 0 if the muon is from the first W, 1 if from the second W
# had_w_indices is 0 if the hadronic W is the first W, 1 if the second W

w_mu_indices = np.array(w_mu_indices)
sl_mask = matched_mu_events.process_id == 1100
sl_w_mu_indices = w_mu_indices[sl_mask]

invert_indices = lambda x: 1 if x == 0 else 0
had_w_indices = []
had_w_indices = [invert_indices(i) for i in sl_w_mu_indices]
# these indices now suit to the matched_mu_sl events

# now look at the hadronic decaying W and check where the fake tau is coming from
# check if W children ( = quarks) are one fatjet

had_w_children_eta = matched_mu_sl.gen_top_w_children_eta[np.arange(len(had_w_indices)), had_w_indices]
had_w_children_phi = matched_mu_sl.gen_top_w_children_phi[np.arange(len(had_w_indices)), had_w_indices]

delr_qq = deltaR(
    had_w_children_eta[:, 0],
    had_w_children_phi[:, 0],
    had_w_children_eta[:, 1],
    had_w_children_phi[:, 1],
)

# tau matching: match to qq fatjet if existent, otherwise match to two quarks
had_w_eta = matched_mu_sl.gen_top_w_eta[np.arange(len(had_w_indices)), had_w_indices]
had_w_phi = matched_mu_sl.gen_top_w_phi[np.arange(len(had_w_indices)), had_w_indices]

tau_delrs = []
tau_matches = [] # 0 if fatjet, 1 if two jets an first matches better, 2 if two jets and second matches better
nb_of_taus = np.ones(len(matched_mu_sl), dtype=np.int_) # // 10
multiple_matches_indices = []
for i in range(len(matched_mu_sl)): # // 10
    if delr_qq[i] < 0.6: # see both q as one fatjet
        delr = deltaR(
                      had_w_eta[i],
                      had_w_phi[i],
                      matched_mu_sl.tau_eta[i],
                      matched_mu_sl.tau_phi[i],
                      )
        tau_delrs.append(np.array(delr))
        tau_matches.append([0])
        nb_of_taus[i] = len(delr)
        if len(delr) > 1:
            multiple_matches_indices.append(i)

    else: # treat both quarks separately
        delr1 = deltaR(
                      had_w_children_eta[i, 0],
                      had_w_children_phi[i, 0],
                      matched_mu_sl.tau_eta[i],
                      matched_mu_sl.tau_phi[i],
                      )
        delr2 = deltaR(
                      had_w_children_eta[i, 1],
                      had_w_children_phi[i, 1],
                      matched_mu_sl.tau_eta[i],
                      matched_mu_sl.tau_phi[i],
                      )
        # loop bc sometimes we have more than one tau
        multiple_tau_delrs = []
        multiple_tau_matches = []
        for j in range(len(delr1)):# loop through all taus
            if delr1[j] < delr2[j]:
                multiple_tau_delrs.append(delr1[j]) # append best matching delr for tau j
                multiple_tau_matches.append(1)# tau j matches to quark 1
            else:
                multiple_tau_delrs.append(delr2[j])
                multiple_tau_matches.append(2)# tau j matches to quark 2
        tau_delrs.append(multiple_tau_delrs)
        tau_matches.append(multiple_tau_matches)
        nb_of_taus[i] = len(delr1)
        if len(delr1) > 1:
            multiple_matches_indices.append(i)

tau_delrs = ak.Array(tau_delrs)
tau_matches = ak.Array(tau_matches)
print("tau_delrs:", tau_delrs)
print("tau_matches:", tau_matches)
##########################################################################

# plot tau delR distribution
taudelr_max = 4.5
taudelr_nbins = 100

taudelr = Hist(hist.axis.Regular(taudelr_nbins, 0, taudelr_max, name="", label="delta R"))
taudelr.fill(ak.flatten(tau_delrs))

x = np.linspace(0, taudelr_max, taudelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
# xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
fig = plt.figure(figsize=(10, 6))
plt.bar(x, taudelr.values(), width=(taudelr_max)/taudelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
# plt.xticks(xticks)
plt.xlabel(r"delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of reconstructed fake tau and best matching children of hadronic decaying gen W, sl mutau channel")
plt.legend()
plt.savefig(f"analysis_mutau/alldelrs_tau_distribution_sl_allevents", dpi=300, bbox_inches='tight')
plt.show()
taudelr.reset()

##########################################################################
# split events in matched (delr < delr_cut) and unmatched
mask_matched = tau_delrs < delr_cut_tau
nb_of_matches = ak.sum(mask_matched, axis=1)

nomatched1fake = matched_mu_sl[(nb_of_matches == 0)& (nb_of_taus == 1)]
nomatched23fake = matched_mu_sl[(nb_of_matches == 0)& (nb_of_taus >= 2)]
onematched1fake = matched_mu_sl[(nb_of_matches == 1) & (nb_of_taus == 1)]
onematched23fake = matched_mu_sl[(nb_of_matches == 1) & (nb_of_taus >= 2)]
twomatched23fake = matched_mu_sl[(nb_of_matches == 2) & (nb_of_taus >= 2)]

# hists
func=logit
lower_border = -14# set to 0 for lin scale
upper_border = 7# set to 1 for lin scale
fake_tau_nbins = 80
nomatched1 = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"One fake $\tau$ in event, none genmatched"))
nomatched23 = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"2-3 fake $\tau$ in event, none genmatched"))
onematched1 = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"One fake $\tau$ in event, one genmatched"))
onematched23 = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"2-3 fake $\tau$ in event, one genmatched"))
twomatched23 = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"2-3 fake $\tau$ in event, two genmatched"))
all_ev          = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="all events"))

nomatched1.fill(func(nomatched1fake.run3_dnn_moe_hh), weight =nomatched1fake.event_weight)
nomatched23.fill(func(nomatched23fake.run3_dnn_moe_hh), weight =nomatched23fake.event_weight)
onematched1.fill(func(onematched1fake.run3_dnn_moe_hh), weight =onematched1fake.event_weight)
onematched23.fill(func(onematched23fake.run3_dnn_moe_hh), weight =onematched23fake.event_weight)
twomatched23.fill(func(twomatched23fake.run3_dnn_moe_hh), weight =twomatched23fake.event_weight)
all_ev.fill(func(matched_mu_sl.run3_dnn_moe_hh), weight =matched_mu_sl.event_weight)

# plot
x = np.linspace(lower_border, upper_border, fake_tau_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, nomatched1.values(), alpha=0.9, label=r"One fake $\tau$ in event, none genmatched", color='green')
ax1.step(x, nomatched23.values(), alpha=0.9, label=r"2-3 fake $\tau$ in event, none genmatched", color='blue')
ax1.step(x, onematched1.values(), alpha=0.9, label=r"One fake $\tau$ in event, one genmatched", color='tab:pink')
ax1.step(x, onematched23.values(), alpha=0.9, label=r"2-3 fake $\tau$ in event, one genmatched", color='tab:orange')
ax1.step(x, twomatched23.values(), alpha=0.9, label=r"2-3 fake $\tau$ in event, two genmatched", color='tab:purple')
ax1.step(x, all_ev.values(), alpha=0.9, label=r'all events', color='red')

ax1.tick_params(axis='y', labelcolor='black')
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(fr"HH output node; tt background events, sl decay, mutau channel with one matched muon split in number of $\tau$ fakes and number of $\tau$ fakes matched to gen W children (matching criterion: $\Delta R < ${delr_cut_tau})", wrap=True)
plt.savefig("analysis_mutau/dnn_tau_matching_sl_allevents", dpi=300, bbox_inches='tight')
plt.show()

nomatched1.reset()
nomatched23.reset()
onematched1.reset()
onematched23.reset()
twomatched23.reset()
all_ev.reset()
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
# di-leptonic channel
# here, reco taus can actually be taus

# find the indices of the W not decaying into a mu, which is the one leading to the (fake) tau
# w_mu_indices is 0 if the muon is from the first W, 1 if from the second W
# had_w_indices is 0 if the hadronic W is the first W, 1 if the second W
matched_mu_dl = matched_mu_events[matched_mu_events.process_id == 1200]


w_mu_indices = np.array(w_mu_indices)
dl_mask = matched_mu_events.process_id == 1200
dl_w_mu_indices = w_mu_indices[dl_mask]

tau_w_indices = []
tau_w_indices = [invert_indices(i) for i in dl_w_mu_indices]

tau_delrs_dl = []
tau_matches = [] # 0 if first children, 1 if second children
nb_of_taus_dl = np.ones(len(matched_mu_dl), dtype=np.int_) # // 10
multiple_matches_indices = []

# find the W which leads to (fake) tau
lep_w_children_eta = matched_mu_dl.gen_top_w_children_eta[np.arange(len(tau_w_indices)), tau_w_indices]
lep_w_children_phi = matched_mu_dl.gen_top_w_children_phi[np.arange(len(tau_w_indices)), tau_w_indices]


for i in range(len(matched_mu_dl)): # // 10
    delr1 = deltaR(
                    lep_w_children_eta[i, 0],
                    lep_w_children_phi[i, 0],
                    matched_mu_dl.tau_eta[i],
                    matched_mu_dl.tau_phi[i],
                    )
    delr2 = deltaR(
                    lep_w_children_eta[i, 1],
                    lep_w_children_phi[i, 1],
                    matched_mu_dl.tau_eta[i],
                    matched_mu_dl.tau_phi[i],
                    )
    multiple_tau_delrs_dl = []
    multiple_tau_matches_dl = []
    for j in range(len(delr1)):
        if delr1[j] < delr2[j]:
            multiple_tau_delrs_dl.append(delr1[j])
            multiple_tau_matches_dl.append(0)
        else:
            multiple_tau_delrs_dl.append(delr2[j])
            multiple_tau_matches_dl.append(1)
    tau_delrs_dl.append(multiple_tau_delrs_dl)
    tau_matches.append(multiple_tau_matches_dl)
    nb_of_taus_dl[i] = len(delr1)
    if len(delr1) > 1:
        multiple_matches_indices.append(i)

tau_delrs_dl = ak.Array(tau_delrs_dl)
tau_matches = ak.Array(tau_matches)
print("dl tau_delrs_dl:", tau_delrs_dl)
print("dl tau_matches:", tau_matches)

# # plot tau delR distribution
# taudelr_max = 4.5
# taudelr_nbins = 100

# taudelr_dl = Hist(hist.axis.Regular(taudelr_nbins, 0, taudelr_max, name="", label="delta R"))
# taudelr_dl.fill(ak.flatten(tau_delrs_dl))

# x = np.linspace(0, taudelr_max, taudelr_nbins + 1)  # bin edges
# x = (x[:-1] + x[1:]) / 2  # bin centers
# # xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
# fig = plt.figure(figsize=(10, 6))
# plt.bar(x, taudelr_dl.values(), width=(taudelr_max)/taudelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
# plt.xticks(xticks)
# plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
# plt.ylabel("Number of events")
# plt.title("Delta R of reconstructed fake tau and best matching children of the non-muonic decaying gen W, dl mutau channel")
# plt.legend()
# plt.savefig(f"analysis_mutau/alldelrs_tau_distribution_dl", dpi=300, bbox_inches='tight')
# plt.show()
# taudelr_dl.reset()
##########################################################################
# split events in matched (delr < delr_cut) and unmatched
# important pdgids: e: 11, mu: 13, tau: 15, nu_e: 12, nu_mu: 14, nu_tau: 16
mask_matched_dl = tau_delrs_dl < delr_cut_tau
nb_of_matches_dl = ak.sum(mask_matched_dl, axis=-1)

mask_tau_event = ak.any(ak.any(abs(matched_mu_dl.gen_top_w_children_pdgId) == 15, axis=2), axis=1)
mask_e_event   = ak.any(ak.any(abs(matched_mu_dl.gen_top_w_children_pdgId) == 11, axis=2), axis=1)
mask_mu_event  = ak.any(ak.any(abs(matched_mu_dl.gen_top_w_children_pdgId) == 13, axis=2), axis=1)
mask_two_mus   = ak.all(ak.any(abs(matched_mu_dl.gen_top_w_children_pdgId) == 13, axis=2), axis=1) # both Ws decay into mus
mask_0_matches = (nb_of_matches_dl == 0)


mutau_0f              = matched_mu_dl[mask_mu_event & mask_tau_event & (nb_of_taus_dl == 1)] # 1 mu, 1 tau, 0 fakes (~90% of data)
mutau_12f             = matched_mu_dl[mask_mu_event & mask_tau_event & (nb_of_taus_dl >= 2)] # 1 mu, 1 tau, 1 fake (few or no events with 2 fakes. Most events here only have match for real tau)
mumu1f_matched        = matched_mu_dl[mask_two_mus & (nb_of_matches_dl == 0)] # always one fake
mumu1f_unmatched      = matched_mu_dl[mask_two_mus& (nb_of_matches_dl == 0)] # always one fake
emu1f                 = matched_mu_dl[mask_e_event & mask_mu_event & (nb_of_taus_dl == 1)]
emu23f                = matched_mu_dl[mask_e_event & mask_mu_event & (nb_of_taus_dl >= 2)]

##########################################################################
# hists
# func=logit
lower_border = -14# set to 0 for lin scale
upper_border = 7# set to 1 for lin scale
fake_tau_nbins = 80
mutau_0f_dl         = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"$\mu\tau$ decay, no $\tau$ fake"))
mutau_12f_dl        = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"$\mu\tau$ decay, 1-2 $\tau$ fakes"))
mumu1f_matched_dl   = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"$\mu\mu$ decay, one $\tau$ fake (matched)"))
mumu1f_unmatched_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"$\mu\mu$ decay, one $\tau$ fake (unmatched)"))
emu1f_dl            = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"$e\mu$ decay, one $\tau$ fake"))
emu23f_dl           = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label=r"$e\mu$ decay, 2-3 $\tau$ fakes"))
all_dl              = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="all events"))

mutau_0f_dl.fill(func(mutau_0f.run3_dnn_moe_hh), weight =mutau_0f.event_weight)
mutau_12f_dl.fill(func(mutau_12f.run3_dnn_moe_hh), weight =mutau_12f.event_weight)
mumu1f_matched_dl.fill(func(mumu1f_matched.run3_dnn_moe_hh), weight =mumu1f_matched.event_weight)
mumu1f_unmatched_dl.fill(func(mumu1f_unmatched.run3_dnn_moe_hh), weight =mumu1f_unmatched.event_weight)
emu1f_dl.fill(func(emu1f.run3_dnn_moe_hh), weight =emu1f.event_weight)
emu23f_dl.fill(func(emu23f.run3_dnn_moe_hh), weight =emu23f.event_weight)
all_dl.fill(func(matched_mu_dl.run3_dnn_moe_hh), weight =matched_mu_dl.event_weight)

# plot
x = np.linspace(lower_border, upper_border, fake_tau_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, mutau_0f_dl.values(), alpha=0.9, label=r"$\mu\tau$ decay, no $\tau$ fake", color='green')
ax1.step(x, mutau_12f_dl.values(), alpha=0.9, label=r"$\mu\tau$ decay, 1-2 $\tau$ fakes", color='blue')
ax1.step(x, mumu1f_matched_dl.values(), alpha=0.9, label=r"$\mu\mu$ decay, one $\tau$ fake (matched)", color='tab:pink')
ax1.step(x, mumu1f_unmatched_dl.values(), alpha=0.9, label=r"$\mu\mu$ decay, one $\tau$ fake (unmatched)", color='tab:orange')
ax1.step(x, emu1f_dl.values(), alpha=0.9, label=r"$e\mu$ decay, one $\tau$ fake", color='tab:purple')
ax1.step(x, emu23f_dl.values(), alpha=0.9, label=r"$e\mu$ decay, 2-3 $\tau$ fakes", color='tab:brown')
ax1.step(x, all_dl.values(), alpha=0.9, label=r'all events', color='red')

ax1.tick_params(axis='y', labelcolor='black')
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(fr"HH output node; tt background events, dl decay, mutau channel with one matched muon split in nb of real $\tau$, number of $\tau$ fakes and number of fake $\tau$ matched to gen W children (matching criterion: $\Delta R < ${delr_cut_tau})", wrap=True)
plt.savefig("analysis_mutau/dnn_tau_matching_dl_allevents", dpi=300, bbox_inches='tight')
plt.show()

mutau_0f_dl.reset()
mutau_12f_dl.reset()
mumu1f_matched_dl.reset()
mumu1f_unmatched_dl.reset()
emu1f_dl.reset()
emu23f_dl.reset()
all_dl.reset()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# check angular distributions of W children
# dl case
lep_w_children_eta = matched_mu_dl.gen_top_w_children_eta[np.arange(len(tau_w_indices)), tau_w_indices]
lep_w_children_phi = matched_mu_dl.gen_top_w_children_phi[np.arange(len(tau_w_indices)), tau_w_indices]
# sl case
had_w_children_eta = matched_mu_sl.gen_top_w_children_eta[np.arange(len(had_w_indices)), had_w_indices]
had_w_children_phi = matched_mu_sl.gen_top_w_children_phi[np.arange(len(had_w_indices)), had_w_indices]

# check angular distributions of the two W's
# dl case
dl_w_eta = matched_mu_dl.gen_top_w_eta
dl_w_phi = matched_mu_dl.gen_top_w_phi
# sl case
sl_w_eta = matched_mu_sl.gen_top_w_eta
sl_w_phi = matched_mu_sl.gen_top_w_phi


delr_qq = deltaR(
    had_w_children_eta[:, 0],
    had_w_children_phi[:, 0],
    had_w_children_eta[:, 1],
    had_w_children_phi[:, 1],
)
delr_ll = deltaR(
    lep_w_children_eta[:, 0],
    lep_w_children_phi[:, 0],
    lep_w_children_eta[:, 1],
    lep_w_children_phi[:, 1],
)

# plot
deltar_qq = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R of had decaying W (sl events)"))
deltar_ll = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R of lep decaying W (dl events)"))
deltar_qq.fill(delr_qq)
deltar_ll.fill(delr_ll)

func=identity
x = np.linspace(0, alldelr_max, alldelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.step(x, deltar_qq.values(), alpha=0.9, label=r"delta R of had decaying W (sl events)", color='green')
ax1.step(x, deltar_ll.values(), alpha=0.9, label=r"delta R of lep decaying W (dl events)", color='red')
ax1.set_yscale("linear")
ax1.set_xscale("linear")
ax1.set_xlabel(r'$\Delta$ R = $\sqrt{\Delta \eta² + \Delta \phi²}$')
ax1.set_ylabel('Number of events', color="black")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
ax1.get_legend_handles_labels()
plt.legend()
plt.title("Events in mutau channel: Delta R of W children for the W not decaying into the matched muon")
plt.savefig("analysis_mutau/delrs_w_children_allevents", dpi=300, bbox_inches='tight')
plt.show()

###################################################################################################################################
# check angular distributions of the two W's
# dl case
dl_w_eta = matched_mu_dl.gen_top_w_eta
dl_w_phi = matched_mu_dl.gen_top_w_phi
# sl case
sl_w_eta = matched_mu_sl.gen_top_w_eta
sl_w_phi = matched_mu_sl.gen_top_w_phi

delr_dl = deltaR(
    dl_w_eta[:, 0],
    dl_w_phi[:, 0],
    dl_w_eta[:, 1],
    dl_w_phi[:, 1],
)
delr_sl = deltaR(
    sl_w_eta[:, 0],
    sl_w_phi[:, 0],
    sl_w_eta[:, 1],
    sl_w_phi[:, 1],
)

# plot
deltar_dl = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R of had decaying W (sl events)"))
deltar_sl = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R of lep decaying W (dl events)"))
deltar_dl.fill(delr_dl)
deltar_sl.fill(delr_sl)

func=identity
x = np.linspace(0, alldelr_max, alldelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.step(x, deltar_dl.values(), alpha=0.9, label=r"delta R of W's in dl events", color='green')
ax1.step(x, deltar_sl.values(), alpha=0.9, label=r"delta R of W's in sl events", color='red')
ax1.set_yscale("linear")
ax1.set_xscale("linear")
ax1.set_xlabel(r'$\Delta$ R = $\sqrt{\Delta \eta² + \Delta \phi²}$')
ax1.set_ylabel('Number of events', color="black")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
ax1.get_legend_handles_labels()
plt.legend()
plt.title("Events in mutau channel: Delta R of gen top W's")
plt.savefig("analysis_mutau/delrs_ws_allevents", dpi=300, bbox_inches='tight')
plt.show()
###################################################################################################################################
# check angular distributions of the two b's and t's
b_eta = matched_mu_events.gen_top_b_eta
b_phi = matched_mu_events.gen_top_b_phi

# check angular distributions of the two tops
top_eta = matched_mu_events.gen_top_t_eta
top_phi = matched_mu_events.gen_top_t_eta

delr_bs = deltaR(
    b_eta[:, 0],
    b_phi[:, 0],
    b_eta[:, 1],
    b_phi[:, 1],
)
delr_ts = deltaR(
    top_eta[:, 0],
    top_phi[:, 0],
    top_eta[:, 1],
    top_phi[:, 1],
)
deltar_bs = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R of had decaying W (sl events)"))
deltar_ts = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R of lep decaying W (dl events)"))
deltar_bs.fill(delr_bs)
deltar_ts.fill(delr_ts)

func=identity
x = np.linspace(0, alldelr_max, alldelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.step(x, deltar_bs.values(), alpha=0.9, label=r"delta R of b's in dl events", color='green')
ax1.set_yscale("linear")
ax1.set_xscale("linear")
ax1.set_xlabel(r'$\Delta$ R = $\sqrt{\Delta \eta² + \Delta \phi²}$')
ax1.set_ylabel('Number of events', color="black")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
ax1.get_legend_handles_labels()
# plt.legend()
plt.title("Events in mutau channel: Delta R of gen b quarks")
plt.savefig("analysis_mutau/delrs_bs_allevents", dpi=300, bbox_inches='tight')
plt.show()

x = np.linspace(0, 10000, alldelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.step(x, deltar_ts.values(), alpha=0.9, label=r"delta R of t's in dl events", color='green')
ax1.set_yscale("linear")
ax1.set_xscale("linear")
ax1.set_xlabel(r'$\Delta$ R = $\sqrt{\Delta \eta² + \Delta \phi²}$')
ax1.set_ylabel('Number of events', color="black")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
ax1.get_legend_handles_labels()

# plt.legend()
plt.title("Events in mutau channel: Delta R of gen top quarks")
plt.savefig("analysis_mutau/delrs_ts_allevents", dpi=300, bbox_inches='tight')
plt.show()

###################################################################################################################################
# met_phi, met_pt
# plot
met_pt_dl = Hist(hist.axis.Regular(alldelr_nbins, 0, 400, name="", label="delta R of had decaying W (sl events)"))
met_pt_sl = Hist(hist.axis.Regular(alldelr_nbins, 0, 400, name="", label="delta R of had decaying W (sl events)"))
met_pt_dl.fill(matched_mu_dl.met_pt)
met_pt_sl.fill(matched_mu_sl.met_pt)

func=identity
x = np.linspace(0, 1000, alldelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.step(x, met_pt_dl.values(), alpha=0.9, label=r"di-leptonic", color='green')
ax1.step(x, met_pt_sl.values(), alpha=0.9, label=r"semi-leptonic", color='red')
ax1.set_yscale("linear")
ax1.set_xscale("linear")
ax1.set_xlabel(r'met $p_T$')
ax1.set_ylabel('Number of events', color="black")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.legend()
plt.title("Events in mutau channel: MET of muon-matched events")
plt.savefig("analysis_mutau/met_allevents", dpi=300, bbox_inches='tight')
plt.show()
