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
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/tt_22pre_v14.parquet")
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
# events_tt = events_tt[events_tt.channel_id == 2] # mutau channel
events_tt = events_tt[ak.any(events_tt.category_ids == 179, axis = 1)] # mutau, res2b channel
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
mask_first_w_matched =  ((delta_rs[:,0,0] < delr_cut_mu) & (abs(pdgids[:,0,0])== 13)) | ((delta_rs[:,0,1] < delr_cut_mu) & (abs(pdgids[:,0,1])== 13))
mask_second_w_matched = ((delta_rs[:,1,0] < delr_cut_mu) & (abs(pdgids[:,1,0])== 13)) | ((delta_rs[:,1,1] < delr_cut_mu) & (abs(pdgids[:,1,1])== 13))

print("done with muon matching")

matched_mu_events = events_tt_train[(mask_first_w_matched) | (mask_second_w_matched)]
fake_mu_events = events_tt_train[(~mask_first_w_matched) & (~mask_second_w_matched)]

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
all_events.fill(func(events_tt_train.run3_dnn_moe_hh), weight =events_tt_train.event_weight)

# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, list(mu_events_dl.values()), alpha=0.9, label=r'matched dl mu events', color='green')
ax1.step(x, list(mu_fakes_dl.values()), alpha=0.9, label=r'fake dl mu events', color='limegreen')
ax1.step(x, list(mu_events_sl.values()), alpha=0.9, label=r'matched sl mu events', color='purple')
ax1.step(x, list(mu_fakes_sl.values()), alpha=0.9, label=r'fake sl mu events', color='darkslateblue')
ax1.step(x, list(mu_fakes_fh.values()), alpha=0.9, label=r'fake fh mu events', color='darkorange')

ax1.step(x, list(all_events.values()), label='all events in mutau, res2b category', color='blue')
# ax1.fill_between(x, list(all_events.values()), color='red', alpha=0.1)

ax1.tick_params(axis='y', labelcolor="black")
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
ax1.legend(loc='upper right')
fig.tight_layout()
plt.title(r"HH output node; tt bg of mutau, res2b category split in correctly matched and fake muon events (matching criterion: $\Delta R <$"+f" {delr_cut_mu})")
plt.savefig("analysis_mutau/res2b_dnn_mu_matching", dpi=300, bbox_inches='tight')
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
sl_mask = events_tt_train.process_id == 1100
events_sl = events_tt_train[sl_mask] # len: all sl events

# now look at the hadronic decaying W and check where the fake tau is coming from
# check if W children ( = quarks) are one fatjet
had_mask = abs(events_sl.gen_top_w_children_pdgId) < 7 # particle level
had_w_child_eta = ak.flatten(events_sl.gen_top_w_children_eta[had_mask], axis=-1)
had_w_child_phi = ak.flatten(events_sl.gen_top_w_children_phi[had_mask], axis=-1)

delr_qq = deltaR(
    had_w_child_eta[:, 0],
    had_w_child_phi[:, 0],
    had_w_child_eta[:, 1],
    had_w_child_phi[:, 1],
)

# tau matching: match to qq fatjet if existent, otherwise match to two quarks
events_fatjet = events_sl[delr_qq < 0.6]
events_2jets  = events_sl[delr_qq >= 0.6]

# slice had decay
had_mask = abs(events_2jets.gen_top_w_children_pdgId) < 7 # particle level
had_w_child_eta = ak.flatten(events_2jets.gen_top_w_children_eta[had_mask], axis=-1)
had_w_child_phi = ak.flatten(events_2jets.gen_top_w_children_phi[had_mask], axis=-1)
lep_w_child_eta = ak.flatten(events_2jets.gen_top_w_children_eta[~had_mask], axis=-1)
lep_w_child_phi = ak.flatten(events_2jets.gen_top_w_children_phi[~had_mask], axis=-1)

had_mask2 = ak.any(abs(events_fatjet.gen_top_w_children_pdgId) < 7, axis=-1) # W level
had_w_eta = events_fatjet.gen_top_w_eta[had_mask2]
had_w_phi = events_fatjet.gen_top_w_phi[had_mask2]
lep_w_eta = events_fatjet.gen_top_w_eta[~had_mask2]
lep_w_phi = events_fatjet.gen_top_w_phi[~had_mask2]

tau_delrs_fatjet = deltaR(
                    had_w_eta[:],
                    had_w_phi[:],
                    events_fatjet.tau_eta[:],
                    events_fatjet.tau_phi[:],
                    )
tau_deltalrs_lep_fatjet1 = deltaR(
                    lep_w_eta[:, 0],
                    lep_w_phi[:, 0],
                    events_fatjet.tau_eta[:],
                    events_fatjet.tau_phi[:],
                    )
tau_deltalrs_lep_fatjet2 = deltaR(
                    lep_w_eta[:, 0],
                    lep_w_phi[:, 0],
                    events_fatjet.tau_eta[:],
                    events_fatjet.tau_phi[:],
                    )

tau_delrs_lep_fatjet = np.minimum(tau_deltalrs_lep_fatjet1, tau_deltalrs_lep_fatjet2)
# events with single qs
delr1 = deltaR(
                had_w_child_eta[:, 0],
                had_w_child_phi[:, 0],
                events_2jets.tau_eta[:],
                events_2jets.tau_phi[:],
                )
delr2 = deltaR(
                had_w_child_eta[:, 1],
                had_w_child_phi[:, 1],
                events_2jets.tau_eta[:],
                events_2jets.tau_phi[:],
                )
delr_qq = np.minimum(delr1, delr1)
delr_lep1 = deltaR(
                    lep_w_child_eta[:, 0],
                    lep_w_child_phi[:, 0],
                    events_2jets.tau_eta[:],
                    events_2jets.tau_phi[:],
                    )
delr_lep2 = deltaR(
                    lep_w_child_eta[:, 1],
                    lep_w_child_phi[:, 1],
                    events_2jets.tau_eta[:],
                    events_2jets.tau_phi[:],
                    )

delr_ll = np.minimum(delr_lep1, delr_lep2)

##########################################################################

# plot tau delR distribution
taudelr_max = 2
taudelr_nbins = 100
taudelr_had = Hist(hist.axis.Regular(taudelr_nbins, 0, taudelr_max, name="", label="delta R"))
taudelr_had.fill(ak.flatten(tau_delrs_fatjet)) # alle matches zu fatjet W
taudelr_had.fill(ak.flatten(delr_qq))
taudelr_lep = Hist(hist.axis.Regular(taudelr_nbins, 0, taudelr_max, name="", label="delta R"))
taudelr_lep.fill(ak.flatten(delr_ll))
taudelr_lep.fill(ak.flatten(tau_delrs_lep_fatjet))

x = np.linspace(0, taudelr_max, taudelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
# xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
fig = plt.figure(figsize=(10, 6))
plt.bar(x, taudelr_had.values(), width=(taudelr_max)/taudelr_nbins, bottom=None, fill=False,  color='pink', edgecolor='pink', label="distance to had W children")
plt.bar(x, taudelr_lep.values(), width=(taudelr_max)/taudelr_nbins, bottom=None, fill=False,  color='blue', edgecolor='blue', label="distance to lep W children")
# plt.xticks(xticks)
plt.xlabel(r"delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of reconstructed fake tau and best matching children of hadronic decaying gen W, sl decay, mutau, res2b category")
plt.legend()
plt.savefig(f"analysis_mutau/res2b_alldelrs_tau_distribution_sl_updated", dpi=300, bbox_inches='tight')
plt.show()
taudelr_had.reset()
taudelr_lep.reset()
##########################################################################
# events_2jets:
tau_delrs_qq = ak.Array(delr_qq) # all delrs to closest q
tau_delrs_ll = ak.Array(delr_ll) # all delrs to closest lep
# events_fatjet:
tau_delrs_fatjet        = ak.Array(tau_delrs_fatjet)       # all delrs to fatjet W
tau_deltalrs_lep_fatjet = ak.Array(tau_delrs_lep_fatjet)# all delrs to closest W lep child


# add masks
faketau_matchedto_qq = ak.concatenate([events_2jets[ak.flatten(tau_delrs_qq < tau_delrs_ll) & ak.flatten(tau_delrs_qq < delr_cut_tau) ], events_fatjet[ak.flatten(tau_delrs_fatjet < tau_deltalrs_lep_fatjet) & ak.flatten(tau_delrs_fatjet < delr_cut_tau)]])
faketau_matchedto_e  = ak.concatenate([events_2jets[ak.flatten(tau_delrs_ll < tau_delrs_qq ) & ak.flatten(tau_delrs_ll < delr_cut_tau) & ak.any(ak.any(abs(events_2jets.gen_top_w_children_pdgId) == 11, axis=-1), axis=-1)],
                                       events_fatjet[ak.flatten(tau_deltalrs_lep_fatjet < tau_delrs_fatjet) & ak.flatten(tau_deltalrs_lep_fatjet < delr_cut_tau) & ak.any(ak.any(abs(events_fatjet.gen_top_w_children_pdgId) == 11, axis=-1), axis=-1)]])
faketau_matchedto_mu  = ak.concatenate([events_2jets[ak.flatten(tau_delrs_ll < tau_delrs_qq ) & ak.flatten(tau_delrs_ll < delr_cut_tau) & ak.any(ak.any(abs(events_2jets.gen_top_w_children_pdgId) == 13, axis=-1), axis=-1)],
                                       events_fatjet[ak.flatten(tau_deltalrs_lep_fatjet < tau_delrs_fatjet) & ak.flatten(tau_deltalrs_lep_fatjet < delr_cut_tau) & ak.any(ak.any(abs(events_fatjet.gen_top_w_children_pdgId) == 13, axis=-1), axis=-1)]])
realtau_matchedto_tau  = ak.concatenate([events_2jets[ak.flatten(tau_delrs_ll < tau_delrs_qq ) & ak.flatten(tau_delrs_ll < delr_cut_tau) & ak.any(ak.any(abs(events_2jets.gen_top_w_children_pdgId) == 15, axis=-1), axis=-1)],
                                       events_fatjet[ak.flatten(tau_deltalrs_lep_fatjet < tau_delrs_fatjet) & ak.flatten(tau_deltalrs_lep_fatjet < delr_cut_tau) & ak.any(ak.any(abs(events_fatjet.gen_top_w_children_pdgId) == 15, axis=-1), axis=-1)]])
faketau_nomatch = ak.concatenate([events_2jets[ ak.flatten(tau_delrs_qq > delr_cut_tau) &  ak.flatten(tau_delrs_ll > delr_cut_tau) ],
                          events_fatjet[ak.flatten(tau_delrs_fatjet > delr_cut_tau) &  ak.flatten(tau_deltalrs_lep_fatjet > delr_cut_tau)]])

# hists
func=logit
lower_border = -14# set to 0 for lin scale
upper_border = 7# set to 1 for lin scale

q_fakes_tau      = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
e_fakes_tau      = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
mu_fakes_tau     = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
matched_real_tau = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
non_matched_fake = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
all_ev           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))

q_fakes_tau.fill(func(faketau_matchedto_qq.run3_dnn_moe_hh), weight =faketau_matchedto_qq.event_weight)
e_fakes_tau.fill(func(faketau_matchedto_e.run3_dnn_moe_hh), weight =faketau_matchedto_e.event_weight)
mu_fakes_tau.fill(func(faketau_matchedto_mu.run3_dnn_moe_hh), weight =faketau_matchedto_mu.event_weight)
matched_real_tau.fill(func(realtau_matchedto_tau.run3_dnn_moe_hh), weight =realtau_matchedto_tau.event_weight)
non_matched_fake.fill(func(faketau_nomatch.run3_dnn_moe_hh), weight =faketau_nomatch.event_weight)
all_ev.fill(func(events_sl.run3_dnn_moe_hh), weight =events_sl.event_weight)

# plot
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, q_fakes_tau.values(), alpha=0.9, label=r"Fake $\tau$ matched to quark", color='green')
ax1.step(x, e_fakes_tau.values(), alpha=0.9, label=r"Fake $\tau$ matched to e", color='blue')
ax1.step(x, mu_fakes_tau.values(), alpha=0.9, label=r"Fake $\tau$ matched to $\mu$", color='tab:pink')
ax1.step(x, matched_real_tau.values(), alpha=0.9, label=r"real $\tau$, genmatched", color='tab:orange')
ax1.step(x, non_matched_fake.values(), alpha=0.9, label=r"unmatched $\tau$ fake", color='tab:purple')
ax1.step(x, all_ev.values(), alpha=0.9, label=r'all events', color='red')

ax1.tick_params(axis='y', labelcolor='black')
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(fr"HH output node; tt background events, sl decay, mutau, res2b category split in $\tau$ fakes matched/not matched to gen W children (matching criterion: $\Delta R < ${delr_cut_tau})", wrap=True)
plt.savefig("analysis_mutau/res2b_dnn_tau_matching_sl", dpi=300, bbox_inches='tight')
plt.show()

q_fakes_tau.reset()
e_fakes_tau.reset()
mu_fakes_tau.reset()
matched_real_tau.reset()
non_matched_fake.reset()
all_ev.reset()
print("done with tau matching, sl decay")
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
# di-leptonic channel
# here, reco taus can actually be taus

# find the indices of the W not decaying into a mu, which is the one leading to the (fake) tau
# w_mu_indices is 0 if the muon is from the first W, 1 if from the second W
# had_w_indices is 0 if the hadronic W is the first W, 1 if the second W
dl_mask = events_tt.process_id == 1200
events_dl = events_tt[dl_mask] # len: all dl events

# first w
delr1 = deltaR(
                events_dl.gen_top_w_children_eta[:, 0, 0],
                events_dl.gen_top_w_children_phi[:, 0, 0],
                events_dl.tau_eta[:],
                events_dl.tau_phi[:],
                )
delr2 = deltaR(
                events_dl.gen_top_w_children_eta[:, 0, 1],
                events_dl.gen_top_w_children_phi[:, 0, 1],
                events_dl.tau_eta[:],
                events_dl.tau_phi[:],
                )
# second w
delr3 = deltaR(
                events_dl.gen_top_w_children_eta[:, 1, 0],
                events_dl.gen_top_w_children_phi[:, 1, 0],
                events_dl.tau_eta[:],
                events_dl.tau_phi[:],
                )
delr4 = deltaR(
                events_dl.gen_top_w_children_eta[:, 1, 1],
                events_dl.gen_top_w_children_phi[:, 1, 1],
                events_dl.tau_eta[:],
                events_dl.tau_phi[:],
                )


# plot tau delR distribution
taudelr_max = 2
taudelr_nbins = 100

taudelr_dl = Hist(hist.axis.Regular(taudelr_nbins, 0, taudelr_max, name="", label="delta R"))
taudelr_dl.fill(ak.flatten(delr1))
taudelr_dl.fill(ak.flatten(delr2))
taudelr_dl.fill(ak.flatten(delr3))
taudelr_dl.fill(ak.flatten(delr4))

x = np.linspace(0, taudelr_max, taudelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
# xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
fig = plt.figure(figsize=(10, 6))
plt.bar(x, taudelr_dl.values(), width=(taudelr_max)/taudelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
# plt.xticks(xticks)
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of reconstructed tau and all gen W children, dl decay, mutau, res2b category")
plt.legend()
plt.savefig(f"analysis_mutau/res2b_alldelrs_tau_distribution_dl", dpi=300, bbox_inches='tight')
plt.show()
taudelr_dl.reset()
##########################################################################
# split events in matched (delr < delr_cut) and unmatched
# important pdgids: e: 11, mu: 13, tau: 15, nu_e: 12, nu_mu: 14, nu_tau: 16
delrs_w1_children = ak.flatten(ak.Array(np.minimum(delr1, delr1)))
delrs_w2_children = ak.flatten(ak.Array(np.minimum(delr3, delr4)))

mask_tau_event = ak.any(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 15, axis=2), axis=1)
mask_e_event   = ak.any(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 11, axis=2), axis=1)
mask_mu_event  = ak.any(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 13, axis=2), axis=1)
mask_two_mus   = ak.all(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 13, axis=2), axis=1) # both Ws decay into mus
mask_two_taus   = ak.all(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 15, axis=2), axis=1)

# all possibilities of decays subdevided into matched and unmatched tau
# ee decay rarely happens
mutau_matchedtau =   events_dl[mask_mu_event & mask_tau_event &
                             (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 15, axis=-1))) |
                              ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 15, axis=-1))))]
mutau_unmatchedtau = events_dl[mask_mu_event & mask_tau_event &
                             (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children > delr_cut_tau))) |
                              ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children > delr_cut_tau))]
emu_ematched2tau   =   events_dl[mask_e_event & mask_mu_event  &
                               (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 11, axis=-1))) |
                               ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 11, axis=-1))))]
emu_nomatchedtau   =   events_dl[mask_e_event & mask_mu_event  &
                               (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children > delr_cut_tau)) |
                               ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children > delr_cut_tau)))]
mumu_matchedtau    =   events_dl[mask_two_mus  &
                               (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children < delr_cut_tau) ) |
                               ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children < delr_cut_tau) ))]
mumu_unmatchedtau   =  events_dl[mask_two_mus  &
                               (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children > delr_cut_tau) ) |
                               ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children > delr_cut_tau) ))]
tautau_matchedtau   =   events_dl[mask_two_taus  &
                               (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children < delr_cut_tau) ) |
                               ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children < delr_cut_tau) ))]
tautau_unmatchedtau   =   events_dl[mask_two_taus  &
                               (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children > delr_cut_tau) ) |
                               ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children > delr_cut_tau) ))]
etau_ematched2tau   =  events_dl[mask_e_event & mask_tau_event &
                             (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 11, axis=-1))) |
                              ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 11, axis=-1))))]
etau_taumatched2tau =  events_dl[mask_e_event & mask_tau_event &
                             (((delrs_w1_children < delrs_w2_children) & (delrs_w1_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 15, axis=-1))) |
                              ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children < delr_cut_tau) & (ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 15, axis=-1))))]
unknown_dl =       events_dl[(((delrs_w1_children < delrs_w2_children) & (delrs_w1_children > delr_cut_tau)) |
                              ((delrs_w2_children < delrs_w1_children) & (delrs_w2_children > delr_cut_tau)))]
print("done with tau matching, dl decay")
##########################################################################
# hists
# func=logit
lower_border = -14# set to 0 for lin scale
upper_border = 7# set to 1 for lin scale


mutau_matchedtau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
mutau_unmatchedtau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
emu_ematched2tau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
emu_nomatchedtau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
mumu_matchedtau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
mumu_unmatchedtau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
tautau_matchedtau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
tautau_unmatchedtau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
etau_ematched2tau_histt           = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=r""))
etau_taumatched2tau_histt              = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
all_dl              = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label="all events"))

mutau_matchedtau_histt.fill(func(mutau_matchedtau.run3_dnn_moe_hh), weight =mutau_matchedtau.event_weight)
mutau_unmatchedtau_histt.fill(func(mutau_unmatchedtau.run3_dnn_moe_hh), weight =mutau_unmatchedtau.event_weight)
emu_ematched2tau_histt.fill(func(emu_ematched2tau.run3_dnn_moe_hh), weight =emu_ematched2tau.event_weight)
emu_nomatchedtau_histt.fill(func(emu_nomatchedtau.run3_dnn_moe_hh), weight =emu_nomatchedtau.event_weight)
mumu_matchedtau_histt.fill(func(mumu_matchedtau.run3_dnn_moe_hh), weight =mumu_matchedtau.event_weight)
mumu_unmatchedtau_histt.fill(func(mumu_unmatchedtau.run3_dnn_moe_hh), weight =mumu_unmatchedtau.event_weight)
tautau_matchedtau_histt.fill(func(tautau_matchedtau.run3_dnn_moe_hh), weight =tautau_matchedtau.event_weight)
tautau_unmatchedtau_histt.fill(func(tautau_unmatchedtau.run3_dnn_moe_hh), weight =tautau_unmatchedtau.event_weight)
etau_ematched2tau_histt.fill(func(etau_ematched2tau.run3_dnn_moe_hh), weight =etau_ematched2tau.event_weight)
etau_taumatched2tau_histt.fill(func(etau_taumatched2tau.run3_dnn_moe_hh), weight =etau_taumatched2tau.event_weight)
all_dl.fill(func(events_dl.run3_dnn_moe_hh), weight =events_dl.event_weight)

# plot
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, mutau_matchedtau_histt.values(), alpha=0.9, label=r"$\mu\tau$ decay, matched $\tau$", color='green')
ax1.step(x, mutau_unmatchedtau_histt.values(), alpha=0.9, label=r"$\mu\tau$ decay, unmatched $\tau$", color='blue')
ax1.step(x, emu_ematched2tau_histt.values(), alpha=0.9, label=r"$e\mu$ decay, matched e", color='tab:pink')
ax1.step(x, emu_nomatchedtau_histt.values(), alpha=0.9, label=r"$e\mu$ decay, unmatched e", color='tab:orange')
ax1.step(x, mumu_matchedtau_histt.values(), alpha=0.9, label=r"$\mu\mu$ decay, matched $\tau$", color='tab:purple')
ax1.step(x, mumu_unmatchedtau_histt.values(), alpha=0.9, label=r"$\mu\mu$ decay, unmatched $\tau$", color='tab:brown')
ax1.step(x, tautau_matchedtau_histt.values(), alpha=0.9, label=r"$\tau\tau$ decay, matched $\tau$", color='tab:olive')
ax1.step(x, tautau_unmatchedtau_histt.values(), alpha=0.9, label=r"$\tau\tau$ decay, unmatched $\tau$", color='midnightblue')
ax1.step(x, etau_ematched2tau_histt.values(), alpha=0.9, label=r"$e\tau$ decay, matched e", color='darkslategrey')
ax1.step(x, etau_taumatched2tau_histt.values(), alpha=0.9, label=r"$e\tau$ decay, unmatched e", color='indigo')
ax1.step(x, all_dl.values(), alpha=0.9, label=r'all events', color='red')

ax1.tick_params(axis='y', labelcolor='black')
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(fr"HH output node; tt background events, dl decay, mutau, res2b category split in different decays and in matched and unmatched reco $\tau$ (matching criterion: $\Delta R < ${delr_cut_tau})", wrap=True)
plt.savefig("analysis_mutau/res2b_dnn_tau_matching_dl", dpi=300, bbox_inches='tight')
plt.show()

mutau_matchedtau_histt.reset()
mutau_unmatchedtau_histt.reset()
emu_ematched2tau_histt.reset()
emu_nomatchedtau_histt.reset()
mumu_matchedtau_histt.reset()
mumu_unmatchedtau_histt.reset()
tautau_matchedtau_histt.reset()
tautau_unmatchedtau_histt.reset()
etau_ematched2tau_histt.reset()
etau_taumatched2tau_histt.reset()
all_dl.reset()
print("done with dl decay plot")
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# origin of gen matched tau
# sl and dl events with tau matches
tau_matched_to_tau_counts = ak.sum(realtau_matchedto_tau.event_weight) + ak.sum(mutau_matchedtau.event_weight) + ak.sum(tautau_matchedtau.event_weight)+ ak.sum(etau_taumatched2tau.event_weight)
# sl, dl events with tau matched to e, mu
tau_matched_to_emu_counts = ak.sum(faketau_matchedto_e.event_weight) + ak.sum(emu_ematched2tau.event_weight) + ak.sum(etau_ematched2tau.event_weight) + ak.sum(faketau_matchedto_mu.event_weight) + ak.sum(mumu_matchedtau.event_weight)
# sl, dl events with tau matched to q
tau_matched_to_q_counts = ak.sum(faketau_matchedto_qq.event_weight)
# sl, dl events with tau unmatched
unknown_counts = ak.sum(faketau_nomatch.event_weight) + ak.sum(unknown_dl.event_weight)
# plot origins
origins = ["Unknown", "jets", r"e/$\mu$", r"$\tau$"]
counts = [unknown_counts, tau_matched_to_q_counts, tau_matched_to_emu_counts, tau_matched_to_tau_counts]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(origins, counts, color='blueviolet', alpha=0.5, edgecolor='black')
plt.xlabel("origin")
plt.ylabel("Number of events")
plt.title(r"Origin of gen matched taus in mutau, res2b category (my matching)")
# plt.xticks(x, labels, rotation=45)
plt.yscale('linear')
plt.savefig("analysis_mutau/res2b_origin_of_genmatched_tau_mutau", dpi=300, bbox_inches='tight')
plt.show()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# plot sl, dl, fh together
# W decay mode
sl_mask = events_tt_train.process_id == 1100
dl_mask = events_tt_train.process_id == 1200
fh_mask = events_tt_train.process_id == 1300
mask_two_taus   = ak.all(ak.any(abs(events_tt_train.gen_top_w_children_pdgId) == 15, axis=2), axis=1) # only relevnt for dl events
events_sl = events_tt_train[sl_mask]
events_dl = events_tt_train[(dl_mask) & (~mask_two_taus)]
events_fh = events_tt_train[fh_mask]
events_tautau = events_tt_train[(dl_mask) & (mask_two_taus)]

for column, label, label2, func, borders in zip([[events_sl.run3_dnn_moe_hh, events_dl.run3_dnn_moe_hh, events_fh.run3_dnn_moe_hh, events_tautau.run3_dnn_moe_hh, events_tt_train.run3_dnn_moe_hh],
                                        [events_sl.ll_mass, events_dl.ll_mass, events_fh.ll_mass, events_tautau.ll_mass, events_tt_train.ll_mass]],#[ak.flatten(events_sl.gen_top_w_mass), ak.flatten(events_dl.gen_top_w_mass), ak.flatten(events_fh.gen_top_w_mass), ak.flatten(events_tautau.gen_top_w_mass), ak.flatten(events_tt_train.gen_top_w_mass)]],
                                        ["HH output node", "lepton mass"],#, "WW mass"],
                                        ["HHdnn", "m_ll"],#, "m_WW"],
                                        [logit, identity],#, identity],
                                        [[-14, 7],[0,150]]):#,[0,160]]):
    sl_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    dl_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    fh_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    tautau_hist       = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    all_hist          = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    sl_hist.fill(func(column[0]), weight =events_sl.event_weight)
    dl_hist.fill(func(column[1]), weight =events_dl.event_weight)
    fh_hist.fill(func(column[2]), weight =events_fh.event_weight)
    tautau_hist.fill(func(column[3]), weight =events_tautau.event_weight)
    all_hist.fill(func(column[4]), weight =events_tt_train.event_weight)

    # plot
    x = np.linspace(borders[0], borders[1], n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Number of events', color="black")
    ax1.step(x, sl_hist.values(), alpha=0.9, label=r"sl decay", color='green')
    ax1.step(x, dl_hist.values(), alpha=0.9, label=r"$e\tau$, $\mu\tau$ dl decay", color='blue')
    ax1.step(x, tautau_hist.values(), alpha=0.9, label=r"$\tau\tau$ decay", color='tab:orange')
    ax1.step(x, fh_hist.values(), alpha=0.9, label=r"fh decay", color='tab:pink')
    ax1.step(x, all_hist.values(), alpha=0.9, label=r"all events", color='red')

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_legend_handles_labels()
    plt.legend()
    ax1.set_yscale("log")
    ax1.set_xscale("linear")
    ax1.set_ylim(bottom=1e-1)
    fig.tight_layout()
    plt.title(fr"{label}; tt background events, mutau, res2b category, split in W decay modes", wrap=True)
    plt.savefig(f"analysis_mutau/{label2}_mutaures2b_W_decay_mode", dpi=300, bbox_inches='tight')
    plt.show()

    sl_hist.reset()
    dl_hist.reset()
    fh_hist.reset()
    tautau_hist.reset()
    all_hist.reset()

# hadronic tau matching
tau_matched_unknown = events_tt[ak.flatten(events_tt.tau_genPartFlav == 0)]
tau_matched_to_e   = events_tt[ak.flatten(events_tt.tau_genPartFlav == 1)]
tau_matched_to_mu  = events_tt[ak.flatten(events_tt.tau_genPartFlav == 2)]
tau_matched_to_tau_emu = ak.concatenate([events_tt[ak.flatten(events_tt.tau_genPartFlav == 3)], events_tt[ak.flatten(events_tt.tau_genPartFlav == 4)]])
tau_matched_to_tau_h = events_tt[ak.flatten(events_tt.tau_genPartFlav == 5)]

for column, label, label2, func, borders in zip([[tau_matched_to_e.run3_dnn_moe_hh, tau_matched_to_mu.run3_dnn_moe_hh, tau_matched_to_tau_h.run3_dnn_moe_hh, tau_matched_to_tau_emu.run3_dnn_moe_hh, tau_matched_unknown.run3_dnn_moe_hh, events_tt.run3_dnn_moe_hh],
                                        [tau_matched_to_e.ll_mass, tau_matched_to_mu.ll_mass, tau_matched_to_tau_h.ll_mass, tau_matched_to_tau_emu.ll_mass, tau_matched_unknown.ll_mass, events_tt.ll_mass]],#[ak.flatten(events_sl.gen_top_w_mass), ak.flatten(events_dl.gen_top_w_mass), ak.flatten(events_fh.gen_top_w_mass), ak.flatten(events_tautau.gen_top_w_mass), ak.flatten(events_tt_train.gen_top_w_mass)]],
                                        ["HH output node", "lepton mass"],#, "WW mass"],
                                        ["HHdnn", "m_ll"],#, "m_WW"],
                                        [logit, identity],#, identity],
                                        [[-14, 7],[0,150]]):#,[0,160]]):
    matched2e_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2mu_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2tauhad_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2tauemu_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2unknown_hist       = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    all_hist          = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))

    matched2e_hist.fill(func(column[0]), weight =tau_matched_to_e.event_weight)
    matched2mu_hist.fill(func(column[1]), weight =tau_matched_to_mu.event_weight)
    matched2tauhad_hist.fill(func(column[2]), weight =tau_matched_to_tau_h.event_weight)
    matched2tauemu_hist.fill(func(column[3]), weight =tau_matched_to_tau_emu.event_weight)
    matched2unknown_hist.fill(func(column[4]), weight =tau_matched_unknown.event_weight)
    all_hist.fill(func(column[5]), weight =events_tt_train.event_weight)

    # plot
    x = np.linspace(borders[0], borders[1], n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Number of events', color="black")
    ax1.step(x, matched2e_hist.values(), alpha=0.9, label=r"reco $\tau$ matched to e", color='green')
    ax1.step(x, matched2mu_hist.values(), alpha=0.9, label=r"reco $\tau$ matched to $\mu$", color='blue')
    ax1.step(x, matched2tauhad_hist.values(), alpha=0.9, label=r"reco $\tau$ matched to $\tau_h$", color='tab:orange')
    ax1.step(x, matched2tauemu_hist.values(), alpha=0.9, label=r"reco $\tau$ matched to $\tau_e$, $\tau_\mu$", color='tab:brown')
    ax1.step(x, matched2unknown_hist.values(), alpha=0.9, label=r"unknown origin of reco $\tau$", color='tab:pink')
    ax1.step(x, all_hist.values(), alpha=0.9, label=r"all events", color='red')

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_legend_handles_labels()
    plt.legend()
    ax1.set_yscale("log")
    ax1.set_xscale("linear")
    ax1.set_ylim(bottom=1e-1)
    fig.tight_layout()
    plt.title(fr"{label}; $\tau$ matching for tt background events, mutau, res2b category", wrap=True)
    plt.savefig(f"analysis_mutau/{label2}_mutaures2b_tau_matching", dpi=300, bbox_inches='tight')
    plt.show()

    matched2e_hist.reset()
    matched2mu_hist.reset()
    matched2tauhad_hist.reset()
    matched2tauemu_hist.reset()
    matched2unknown_hist.reset()
    all_hist.reset()
