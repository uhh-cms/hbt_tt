import itertools

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

eps = 0#1e-6 # set eps=0 for normal scale
lower_border = 0#-14# set to 0 for lin scale
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

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_tt = events_tt[events_tt.channel_id == 2] # mutau channel
# events_tt_train = events_tt
events_tt_train = ak.concatenate([events_tt[:10000], events_tt[844445:854446]]) # first ev are dl, second sl
events_tt_train = ak.concatenate([events_tt_train, events_tt[844127:844444,]]) # also add fh events
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
plt.bar(x, alldelr.values(), width=(alldelr_max)/alldelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
plt.xticks(xticks)
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.show()
alldelr.reset()


##################################################################################################################
# muon matching
# array with all delrs, to not loose the indices
delta_rs1 = np.column_stack([delr1_emu, delr2_emu])
delta_rs2 = np.column_stack([delr3_emu, delr4_emu])
delta_rs = np.stack([delta_rs1, delta_rs2], axis=1)
delta_rs = ak.Array(delta_rs)

# event-wise loop to find events which definitely hae one genmatched mu
# vectorised way is very difficult because indices get lost very easily
matched_mu_indices = []
w_mu_indices = []
flat_delrs = ak.flatten(delta_rs, axis=-1)
flat_pdgids = ak.flatten(events_tt_train.gen_top_w_children_pdgId, axis=-1)
for i in range(len(events_tt_train)):
        for j in range(3):
            if flat_delrs[i][j] < delr_cut_mu and abs(flat_pdgids[i][j]) == 13:
                # print(f"Event {i} has a gen-matched muon with pdgId {flat_pdgids[i][j]} and delta R {flat_delrs[i][j]}")
                matched_mu_indices.append(i)
                if j < 2:
                    w_mu_indices.append(0) # muon from first W
                else:
                    w_mu_indices.append(1) # muon from second W
                pass
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
plt.title(f"HH output node; mutau channel tt bg split in correctly matched and fake muon events (matching criterion: $\Delta R < {delr_cut_mu}$)")
plt.savefig("analysis_mutau/dnn_mu_matching", dpi=300, bbox_inches='tight')
plt.show()

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
nb_of_matched_taus = np.ones(len(matched_mu_sl), dtype=np.int_) # // 10
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
        if len(delr) > 1:
            # print("Warning: delr has more than one entry, check indices!")
            nb_of_matched_taus[i] = len(delr)
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
        multiple_tau_delrs = []
        multiple_tau_matches = []
        for j in range(len(delr1)):
            if delr1[j] < delr2[j]:
                multiple_tau_delrs.append(delr1[j])
                multiple_tau_matches.append(1)
            else:
                multiple_tau_delrs.append(delr2[j])
                multiple_tau_matches.append(2)
        tau_delrs.append(multiple_tau_delrs)
        tau_matches.append(multiple_tau_matches)
        if len(delr1) > 1:
            nb_of_matched_taus[i] = len(delr1)
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
plt.xticks(xticks)
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of reconstructed fake tau and best matching children of hadronic decaying gen W, sl mutau channel")
plt.legend()
plt.savefig(f"analysis_mutau/alldelrs_tau_distribution_sl", dpi=300, bbox_inches='tight')
plt.show()
taudelr.reset()

##########################################################################
# split events in matched (delr < delr_cut) and unmatched
matched_tau_events   = matched_mu_sl[ak.any(tau_delrs < delr_cut_tau, axis=1)]
unmatched_tau_events = matched_mu_sl[ak.all(tau_delrs >= delr_cut_tau, axis=1)]
# split delta rs
matched_tau_delrs    = tau_delrs[ak.any(tau_delrs < delr_cut_tau, axis=1)]
unmatched_tau_delrs  = tau_delrs[ak.all(tau_delrs >= delr_cut_tau, axis=1)]
# split index of match (fatjet/qq match)
matched_tau_matches  = tau_matches[ak.any(tau_delrs < delr_cut_tau, axis=1)]
unmatched_tau_matches  = tau_matches[ak.all(tau_delrs >= delr_cut_tau, axis=1)]
# split nb of matched taus per event
matched_nb_of_matched_taus   = nb_of_matched_taus[ak.any(tau_delrs < delr_cut_tau, axis=1)]
unmatched_nb_of_matched_taus = nb_of_matched_taus[ak.all(tau_delrs >= delr_cut_tau, axis=1)]

# nb of fakes
mask_one_fake = matched_nb_of_matched_taus == 1
matched_tau_one_fake      = matched_tau_events[matched_nb_of_matched_taus == 1]
matched_tau_two_fakes     = matched_tau_events[matched_nb_of_matched_taus == 2]
matched_tau_three_fakes   = matched_tau_events[matched_nb_of_matched_taus == 3]
unmatched_tau_one_fake    = unmatched_tau_events[unmatched_nb_of_matched_taus == 1]
unmatched_tau_two_fakes   = unmatched_tau_events[unmatched_nb_of_matched_taus == 2]
unmatched_tau_three_fakes = unmatched_tau_events[unmatched_nb_of_matched_taus == 3]

##########################################################################
# hists
func=logit
lower_border = -12# set to 0 for lin scale
upper_border = 5# set to 1 for lin scale
fake_tau_nbins = 80
m1f = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="matched event, 1 fake tau"))
m2f = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="matched event, 2 fake taus"))
m3f = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="matched event, 3 fake taus"))
u1f = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="unmatched event, 1 fake tau"))
u2f = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="unmatched event, 2 fake taus"))
u3f = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="unmatched event, 3 fake taus"))
all = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="all events"))

m1f.fill(func(matched_tau_one_fake.run3_dnn_moe_hh), weight =matched_tau_one_fake.event_weight)
m2f.fill(func(matched_tau_two_fakes.run3_dnn_moe_hh), weight =matched_tau_two_fakes.event_weight)
m3f.fill(func(matched_tau_three_fakes.run3_dnn_moe_hh), weight =matched_tau_three_fakes.event_weight)
u1f.fill(func(unmatched_tau_one_fake.run3_dnn_moe_hh), weight =unmatched_tau_one_fake.event_weight)
u2f.fill(func(unmatched_tau_two_fakes.run3_dnn_moe_hh), weight =unmatched_tau_two_fakes.event_weight)
u3f.fill(func(unmatched_tau_three_fakes.run3_dnn_moe_hh), weight =unmatched_tau_three_fakes.event_weight)
all.fill(func(matched_tau_events.run3_dnn_moe_hh), weight =matched_tau_events.event_weight)
all.fill(func(unmatched_tau_events.run3_dnn_moe_hh), weight =unmatched_tau_events.event_weight)

# plot
x = np.linspace(lower_border, upper_border, fake_tau_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, m1f.values(), alpha=0.9, label=r'event with matched tau; 1 fake tau', color='green')
ax1.step(x, m2f.values(), alpha=0.9, label=r'event with matched tau; 2 fake taus', color='blue')
ax1.step(x, m3f.values(), alpha=0.9, label=r'event with matched tau; 3 fake taus', color='tab:pink')
ax1.step(x, u1f.values(), alpha=0.9, label=r'event with no matched tau; 1 fake tau', color='tab:orange')
ax1.step(x, u2f.values(), alpha=0.9, label=r'event with no matched tau; 2 fake taus', color='tab:purple')
ax1.step(x, u3f.values(), alpha=0.9, label=r'event with no matched tau; 3 fake taus', color='tab:brown')
ax1.step(x, all.values(), alpha=0.9, label=r'all events', color='red')

ax1.tick_params(axis='y', labelcolor='black')
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(fr"HH output node; tt background events, sl decay, mutau channel with one matched muon and one (un)matched tau (matching criterion: $\Delta R < ${delr_cut_tau}) split in nb of tau fakes", wrap=True)
plt.savefig("analysis_mutau/dnn_tau_matching_sl", dpi=300, bbox_inches='tight')
plt.show()

m1f.reset()
m2f.reset()
m3f.reset()
u1f.reset()
u2f.reset()
u3f.reset()
all.reset()

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
nb_of_matched_taus = np.ones(len(matched_mu_dl), dtype=np.int_) # // 10
multiple_matches_indices = []

# find the W which leads to (fake) tau
had_w_children_eta = matched_mu_dl.gen_top_w_children_eta[np.arange(len(tau_w_indices)), tau_w_indices]
had_w_children_phi = matched_mu_dl.gen_top_w_children_phi[np.arange(len(tau_w_indices)), tau_w_indices]

for i in range(len(matched_mu_dl)): # // 10 
    delr1 = deltaR(
                    had_w_children_eta[i, 0],
                    had_w_children_phi[i, 0],
                    matched_mu_dl.tau_eta[i],
                    matched_mu_dl.tau_phi[i],
                    )
    delr2 = deltaR(
                    had_w_children_eta[i, 1],
                    had_w_children_phi[i, 1],
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
    if len(delr1) > 1:
        nb_of_matched_taus[i] = len(delr1)
        multiple_matches_indices.append(i)

tau_delrs_dl = ak.Array(tau_delrs_dl)
tau_matches = ak.Array(tau_matches)
print("dl tau_delrs_dl:", tau_delrs_dl)
print("dl tau_matches:", tau_matches)

# plot tau delR distribution 
taudelr_max = 4.5
taudelr_nbins = 100

taudelr_dl = Hist(hist.axis.Regular(taudelr_nbins, 0, taudelr_max, name="", label="delta R"))
taudelr_dl.fill(ak.flatten(tau_delrs_dl))

x = np.linspace(0, taudelr_max, taudelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
# xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
fig = plt.figure(figsize=(10, 6))
plt.bar(x, taudelr_dl.values(), width=(taudelr_max)/taudelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
plt.xticks(xticks)
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of reconstructed fake tau and best matching children of the non-muonic decaying gen W, dl mutau channel")
plt.legend()
plt.savefig(f"analysis_mutau/alldelrs_tau_distribution_dl", dpi=300, bbox_inches='tight')
plt.show()
taudelr_dl.reset()

##########################################################################
# split events in matched (delr < delr_cut) and unmatched
tau_w_indices = ak.Array(tau_w_indices)
matched_tau_events   = matched_mu_dl[ak.any(tau_delrs_dl < delr_cut_tau, axis=1)]
unmatched_tau_events = matched_mu_dl[ak.all(tau_delrs_dl >= delr_cut_tau, axis=1)]
# split W tau indices
matched_tau_w_indices   = tau_w_indices[ak.any(tau_delrs_dl < delr_cut_tau, axis=1)]
unmatched_tau_w_indices = tau_w_indices[ak.all(tau_delrs_dl >= delr_cut_tau, axis=1)]
# split delta rs
matched_tau_delrs    = tau_delrs_dl[ak.any(tau_delrs_dl < delr_cut_tau, axis=1)]
unmatched_tau_delrs  = tau_delrs_dl[ak.all(tau_delrs_dl >= delr_cut_tau, axis=1)]
# split index of match (fatjet/qq match)
matched_tau_matches  = tau_matches[ak.any(tau_delrs_dl < delr_cut_tau, axis=1)]
unmatched_tau_matches  = tau_matches[ak.all(tau_delrs_dl >= delr_cut_tau, axis=1)]
# split nb of matched taus per event
matched_nb_of_matched_taus   = nb_of_matched_taus[ak.any(tau_delrs_dl < delr_cut_tau, axis=1)]
unmatched_nb_of_matched_taus = nb_of_matched_taus[ak.all(tau_delrs_dl >= delr_cut_tau, axis=1)]
#
matched_pdgids   = matched_tau_events.gen_top_w_children_pdgId[np.arange(len(matched_tau_w_indices)), matched_tau_w_indices]
unmatched_pdgids = unmatched_tau_events.gen_top_w_children_pdgId[np.arange(len(unmatched_tau_w_indices)), unmatched_tau_w_indices]

from IPython import embed; embed()

# nb of fakes
# important pdgids: e: 11, mu: 13, tau: 15, nu_e: 12, nu_mu: 14, nu_tau: 16

mask_tau_event = ak.any((matched_pdgids == 15) | (matched_pdgids == -15), axis=-1)
mask_e_event   = ak.any((matched_pdgids == 11) | (matched_pdgids == -11), axis=-1)
mask_mu_event  = ak.any((matched_pdgids == 13) | (matched_pdgids == -13), axis=-1)
mask_e_event_unmatched   = ak.any((unmatched_pdgids == 11) | (unmatched_pdgids == -11), axis=-1)
mask_mu_event_unmatched  = ak.any((unmatched_pdgids == 13) | (unmatched_pdgids == -13), axis=-1)

tau_event_no_fake  = matched_tau_events[(matched_nb_of_matched_taus == 1) & mask_tau_event]
tau_event_12_fakes = matched_tau_events[(matched_nb_of_matched_taus >= 2) & mask_tau_event]
emu_event_matched  =  matched_tau_events[mask_e_event | mask_mu_event]
emu_event_unmatched = unmatched_tau_events[mask_e_event_unmatched | mask_mu_event_unmatched]





# matched_tau_one_fake      = matched_tau_events[matched_nb_of_matched_taus == 1]
# matched_tau_two_fakes     = matched_tau_events[matched_nb_of_matched_taus == 2]
# matched_tau_three_fakes   = matched_tau_events[matched_nb_of_matched_taus == 3]
# unmatched_tau_one_fake    = unmatched_tau_events[unmatched_nb_of_matched_taus == 1]
# unmatched_tau_two_fakes   = unmatched_tau_events[unmatched_nb_of_matched_taus == 2]
# unmatched_tau_three_fakes = unmatched_tau_events[unmatched_nb_of_matched_taus == 3]

##########################################################################
# hists
func=logit
lower_border = -12# set to 0 for lin scale
upper_border = 5# set to 1 for lin scale
fake_tau_nbins = 80
m1f_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="matched event, 1 fake tau"))
m2f_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="matched event, 2 fake taus"))
m3f_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="matched event, 3 fake taus"))
u1f_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="unmatched event, 1 fake tau"))
u2f_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="unmatched event, 2 fake taus"))
u3f_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="unmatched event, 3 fake taus"))
all_dl = Hist(hist.axis.Regular(fake_tau_nbins, lower_border, upper_border, name="", label="all events"))

m1f_dl.fill(func(matched_tau_one_fake.run3_dnn_moe_hh), weight =matched_tau_one_fake.event_weight)
m2f_dl.fill(func(matched_tau_two_fakes.run3_dnn_moe_hh), weight =matched_tau_two_fakes.event_weight)
m3f_dl.fill(func(matched_tau_three_fakes.run3_dnn_moe_hh), weight =matched_tau_three_fakes.event_weight)
u1f_dl.fill(func(unmatched_tau_one_fake.run3_dnn_moe_hh), weight =unmatched_tau_one_fake.event_weight)
u2f_dl.fill(func(unmatched_tau_two_fakes.run3_dnn_moe_hh), weight =unmatched_tau_two_fakes.event_weight)
u3f_dl.fill(func(unmatched_tau_three_fakes.run3_dnn_moe_hh), weight =unmatched_tau_three_fakes.event_weight)
all_dl.fill(func(matched_tau_events.run3_dnn_moe_hh), weight =matched_tau_events.event_weight)
all_dl.fill(func(unmatched_tau_events.run3_dnn_moe_hh), weight =unmatched_tau_events.event_weight)

# plot
x = np.linspace(lower_border, upper_border, fake_tau_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, m1f_dl.values(), alpha=0.9, label=r'event with matched tau; 1 fake tau', color='green')
ax1.step(x, m2f_dl.values(), alpha=0.9, label=r'event with matched tau; 2 fake taus', color='blue')
ax1.step(x, m3f_dl.values(), alpha=0.9, label=r'event with matched tau; 3 fake taus', color='tab:pink')
ax1.step(x, u1f_dl.values(), alpha=0.9, label=r'event with no matched tau; 1 fake tau', color='tab:orange')
ax1.step(x, u2f_dl.values(), alpha=0.9, label=r'event with no matched tau; 2 fake taus', color='tab:purple')
ax1.step(x, u3f_dl.values(), alpha=0.9, label=r'event with no matched tau; 3 fake taus', color='tab:brown')
ax1.step(x, all_dl.values(), alpha=0.9, label=r'all events', color='red')

ax1.tick_params(axis='y', labelcolor='black')
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(fr"HH output node; tt background events, dl decay, mutau channel with one matched muon and one (un)matched tau (matching criterion: $\Delta R < ${delr_cut_tau}) split in nb of tau fakes", wrap=True)
plt.savefig("analysis_mutau/dnn_tau_matching_dl", dpi=300, bbox_inches='tight')
plt.show()

m1f_dl.reset()
m2f_dl.reset()
m3f_dl.reset()
u1f_dl.reset()
u2f_dl.reset()
u3f_dl.reset()
all_dl.reset()