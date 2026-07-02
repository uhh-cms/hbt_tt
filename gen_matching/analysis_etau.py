import itertools

import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt


"""This script analyses the etau channel, res2b category and matches e, tau to gen W children.
"""
n_bins = 100

delr_cut_e = 0.25
delr_cut_b = 0.4

eps = 1e-6 # set eps=0 for normal scale
lower_border = -14# set to 0 for lin scale
upper_border = 8# set to 1 for lin scale
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
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/prod24/hh_22pre_v14.parquet")
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_hh = events_hh[events_hh.run3_dnn_moe_hh > 0]
# events_tt = events_tt[events_tt.channel_id == 1] # mutau channel
events_tt = events_tt[ak.any(events_tt.category_ids == 151, axis = 1)] # etau, res2b channel
events_hh = events_hh[ak.any(events_hh.category_ids == 151, axis = 1)] # etau, res2b channel
events_tt_train = events_tt

# split signal data into the different kappa lambda hypotheses
# for the four options of this dataset, kappa t is always 1
events_kl1  = events_hh[events_hh.process_id == 21101]
events_kl0  = events_hh[events_hh.process_id == 21114]
events_kl245 = events_hh[events_hh.process_id == 21120]
events_kl5  = events_hh[events_hh.process_id == 21121]

def deltaR(eta1, phi1, eta2, phi2):
    delta_eta = eta2 - eta1
    delta_phi = phi2 - phi1
    # Ensure delta_phi is in the range [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

# gen matching: electrons with gen W children
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
mask_e = abs(events_tt_train.gen_top_w_tau_children_pdgId) == 11

# # this is a nested mask, we need to get to the event level
# event_mask = ak.any(mask_e, axis=-1)   # tau children level
# event_mask = ak.any(event_mask, axis=-1)   # tau level
# event_mask = ak.any(event_mask, axis=-1)   # event level

# tau_events_eta = events_tt_train.emu_eta[event_mask]
# tau_events_phi = events_tt_train.emu_phi[event_mask]
# tau_children_events = events_tt_train[event_mask]

# tau_children_eta = tau_children_events.gen_top_w_tau_children_eta[abs(tau_children_events.gen_top_w_tau_children_pdgId) == 11]
# tau_children_phi = events_tt_train.gen_top_w_tau_children_phi[event_mask]
delr_tauwchildren = []

tau_events_e_children_phi = ak.flatten(ak.flatten(events_tt_train.gen_top_w_tau_children_phi[mask_e], axis=2), axis=2)
tau_events_e_children_eta = ak.flatten(ak.flatten(events_tt_train.gen_top_w_tau_children_eta[mask_e], axis=2), axis=2)
nb_of_tau_children = ak.num(tau_events_e_children_phi)
print("number of tau children:", nb_of_tau_children, "Number of events:", len(nb_of_tau_children))
for i in range(len(events_tt_train)):
    if nb_of_tau_children[i] == 0:
        delr_tauwchildren.append(100)
    elif nb_of_tau_children[i] == 1:
        delr_emu_wtau = deltaR(
            events_tt_train.emu_eta[i],
            events_tt_train.emu_phi[i],
            tau_events_e_children_eta[i],
            tau_events_e_children_phi[i]
        )
        delr_tauwchildren.append(delr_emu_wtau[0])
    elif nb_of_tau_children[i] == 2:
        delr1_emu_wtau = deltaR(
            events_tt_train.emu_eta[i],
            events_tt_train.emu_phi[i],
            tau_events_e_children_eta[i, 0],
            tau_events_e_children_phi[i, 0]
        )
        delr2_emu_wtau = deltaR(
            events_tt_train.emu_eta[i],
            events_tt_train.emu_phi[i],
            tau_events_e_children_eta[i, 1],
            tau_events_e_children_phi[i, 1]
        )
        delr_tauwchildren.append(min(delr2_emu_wtau, delr1_emu_wtau)[0])
    elif nb_of_tau_children[i] == 3:
        delr1_emu_wtau = deltaR(
            events_tt_train.emu_eta[i],
            events_tt_train.emu_phi[i],
            tau_events_e_children_eta[i, 0],
            tau_events_e_children_phi[i, 0]
        )
        delr2_emu_wtau = deltaR(
            events_tt_train.emu_eta[i],
            events_tt_train.emu_phi[i],
            tau_events_e_children_eta[i, 1],
            tau_events_e_children_phi[i, 1]
        )
        delr3_emu_wtau = deltaR(
            events_tt_train.emu_eta[i],
            events_tt_train.emu_phi[i],
            tau_events_e_children_eta[i, 2],
            tau_events_e_children_phi[i, 2]
        )
        delr_tauwchildren.append(min(delr2_emu_wtau, delr1_emu_wtau, delr3_emu_wtau)[0])
alldelr_max = 1.5
alldelr_nbins = 100

tau_children_e_delr = ak.Array(delr_tauwchildren)[ak.Array(delr_tauwchildren) < 100]
alldelr = Hist(hist.axis.Regular(alldelr_nbins, 0, alldelr_max, name="", label="delta R"))
alldelr.fill(tau_children_e_delr, weight = events_tt_train[ak.Array(delr_tauwchildren) < 100].event_weight)

x = np.linspace(0, alldelr_max, alldelr_nbins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
# xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
fig = plt.figure(figsize=(10, 6))
plt.bar(x, alldelr.values(), width=(alldelr_max)/alldelr_nbins, bottom=None, fill=True,  color='pink', edgecolor='black')
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of all tau W children electrons with reco electrons")
plt.savefig(f"analysis_etau/tauwchildren_alldelrs_e_distribution", dpi=300, bbox_inches='tight')
# plt.xticks(xticks)
plt.show()
alldelr.reset()
#################################################################################################################
# e matching
# event passes e matching if at least one e is matched
# array with all delrs, to not loose the indices
delta_rs1 = np.column_stack([delr1_emu, delr2_emu])
delta_rs2 = np.column_stack([delr3_emu, delr4_emu])
delta_rs = np.stack([delta_rs1, delta_rs2], axis=1)
delta_rs = ak.Array(delta_rs)

# event-wise loop to find events which definitely have one genmatched e
# vectorised way is very difficult because indices get lost very easily
pdgids = events_tt_train.gen_top_w_children_pdgId
mask_first_w_matched =  ((delta_rs[:,0,0] < delr_cut_e) & (abs(pdgids[:,0,0])== 11)) | ((delta_rs[:,0,1] < delr_cut_e) & (abs(pdgids[:,0,1])== 11))
mask_second_w_matched = ((delta_rs[:,1,0] < delr_cut_e) & (abs(pdgids[:,1,0])== 11)) | ((delta_rs[:,1,1] < delr_cut_e) & (abs(pdgids[:,1,1])== 11))
mask_tau_w_children_matched = (ak.Array(delr_tauwchildren) < delr_cut_e)

print("done with e matching")

matched_e_events = events_tt_train[(mask_first_w_matched) | (mask_second_w_matched)]
matched_e_tauwchildren_events = events_tt_train[mask_tau_w_children_matched]
fake_e_events = events_tt_train[(~mask_first_w_matched) & (~mask_second_w_matched) & (~mask_tau_w_children_matched)]

matched_e_dl = matched_e_events[matched_e_events.process_id == 1200]
matched_e_sl = matched_e_events[matched_e_events.process_id == 1100]
matched_e_fh = matched_e_events[matched_e_events.process_id == 1300]
matched_e_wtauchildren_dl = matched_e_tauwchildren_events[matched_e_tauwchildren_events.process_id == 1200]
matched_e_wtauchildren_sl = matched_e_tauwchildren_events[matched_e_tauwchildren_events.process_id == 1100]
matched_e_wtauchildren_fh = matched_e_tauwchildren_events[matched_e_tauwchildren_events.process_id == 1300]
fake_e_dl = fake_e_events[fake_e_events.process_id == 1200]
fake_e_sl = fake_e_events[fake_e_events.process_id == 1100]
fake_e_fh = fake_e_events[fake_e_events.process_id == 1300]

# initialize hists
# fh match hist not necessary, as all fh events are unmatched (which we expected)
e_events_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e", label="e"))
e_events_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e", label="e"))
e_wtauch_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e", label="e_ch"))
e_wtauch_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e", label="e_ch"))
e_wtauch_fh = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e", label="e_ch"))
e_fakes_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e_fakes", label="e_fakes"))
e_fakes_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e_fakes", label="e_fakes"))
e_fakes_fh = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="e_fakes", label="e_fakes"))
all_events = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="all_events", label="all_events"))
# fill hists
e_events_dl.fill(func(matched_e_dl.run3_dnn_moe_hh), weight =matched_e_dl.event_weight)
e_events_sl.fill(func(matched_e_sl.run3_dnn_moe_hh), weight =matched_e_sl.event_weight)
e_wtauch_dl.fill(func(matched_e_wtauchildren_dl.run3_dnn_moe_hh), weight =matched_e_wtauchildren_dl.event_weight)
e_wtauch_sl.fill(func(matched_e_wtauchildren_sl.run3_dnn_moe_hh), weight =matched_e_wtauchildren_sl.event_weight)
e_wtauch_fh.fill(func(matched_e_wtauchildren_fh.run3_dnn_moe_hh), weight =matched_e_wtauchildren_fh.event_weight)
e_fakes_dl.fill(func(fake_e_dl.run3_dnn_moe_hh), weight =fake_e_dl.event_weight)
e_fakes_sl.fill(func(fake_e_sl.run3_dnn_moe_hh), weight =fake_e_sl.event_weight)
e_fakes_fh.fill(func(fake_e_fh.run3_dnn_moe_hh), weight =fake_e_fh.event_weight)
all_events.fill(func(events_tt_train.run3_dnn_moe_hh), weight =events_tt_train.event_weight)

# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, list(all_events.values()), label='all events in etau, res2b category', color='red')
ax1.step(x, list(e_events_dl.values()), alpha=0.9, label=r'dl, e matched to top W children', color='green')
ax1.step(x, list(e_events_sl.values()), alpha=0.9, label=r'sl, e matched to top W children', color='purple')
ax1.step(x, list(e_wtauch_dl.values()), alpha=0.9, label=r'dl, e matched to tau W children', color='limegreen')
ax1.step(x, list(e_wtauch_sl.values()), alpha=0.9, label=r'sl, e matched to tau W children', color='darkslateblue')
ax1.step(x, list(e_fakes_dl.values()), alpha=0.9, label=r'fake dl e events', color='tab:orange')
ax1.step(x, list(e_fakes_sl.values()), alpha=0.9, label=r'fake sl e events', color='tab:brown')
# ax1.step(x, list(e_wtauch_fh.values()), alpha=0.9, label=r'fh, e matched to tau W children', color='tab:pink')
# ax1.step(x, list(e_fakes_fh.values()), alpha=0.9, label=r'fake fh e events', color='darkorange')

ax1.tick_params(axis='y', labelcolor="black")
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
ax1.legend(loc='upper right')
fig.tight_layout()
plt.title(r"HH output node; tt background events of etau, res2b category split in correctly matched and fake e events (matching criterion: $\Delta R <$"+f" {delr_cut_e})", wrap=True)
plt.savefig("analysis_etau/res2b_dnn_e_matching", dpi=300, bbox_inches='tight')
plt.show()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# plot sl, dl, fh together
# W decay mode
sl_mask = events_tt_train.process_id == 1100
dl_mask = events_tt_train.process_id == 1200
fh_mask = events_tt_train.process_id == 1300

# re-define tau as hadronically deyacing tau
# this is much closer to our analysis

# masks for tau decay:
event_mask_1e =  ak.any(ak.any(ak.any(abs(events_tt_train.gen_top_w_tau_children_pdgId) == 11, axis=-1), axis=-1), axis=-1) & ak.any(ak.any(ak.any(abs(events_tt_train.gen_top_w_tau_children_pdgId) == 12, axis=-1), axis=-1), axis=-1)# checked
event_mask_1mu = ak.any(ak.any(ak.any(abs(events_tt_train.gen_top_w_tau_children_pdgId) == 13, axis=-1), axis=-1), axis=-1) & ak.any(ak.any(ak.any(abs(events_tt_train.gen_top_w_tau_children_pdgId) == 14, axis=-1), axis=-1), axis=-1)# checked
event_mask_2e = ak.all(ak.any(ak.any(abs(events_tt_train.gen_top_w_tau_children_pdgId) == 11, axis=-1), axis=-1), axis=-1) & ak.all(ak.any(ak.any(abs(events_tt_train.gen_top_w_tau_children_pdgId) == 12, axis=-1), axis=-1), axis=-1)# checked
event_mask_2mu = ak.all(ak.any(ak.any(abs(events_tt_train.gen_top_w_tau_children_pdgId) == 13, axis=-1), axis=-1), axis=-1) # checked, empty
event_mask_emu = ak.any(ak.any(ak.any((abs(events_tt_train.gen_top_w_tau_children_pdgId) == 11) & (abs(events_tt_train.gen_top_w_tau_children_pdgId) == 13), axis=-1), axis=-1), axis=-1)# checked, empty

mask_1leptonic_tau_decay = (event_mask_1e | event_mask_1mu) & (~event_mask_2e)
mask_2leptonic_tau_decay = event_mask_2e | event_mask_2mu | event_mask_emu
mask_gen_tau = ak.any(ak.any(abs(events_tt_train.gen_top_w_children_pdgId) == 15, axis=2),axis=1)
mask_had_tau = mask_gen_tau & ~mask_1leptonic_tau_decay
mask_two_tauhad   = ak.all(ak.any(abs(events_tt_train.gen_top_w_children_pdgId) == 15, axis=2), axis=1) & (~mask_2leptonic_tau_decay) & (~mask_1leptonic_tau_decay) # checked

events_sl_notau = events_tt_train[sl_mask & (~mask_had_tau)]# checked
events_sl_1tau  = events_tt_train[sl_mask & mask_had_tau]# checked
events_dl_onetau = events_tt_train[(dl_mask) & (~mask_two_tauhad)]# checked
events_fh = events_tt_train[fh_mask]# checked
events_dl_tautau = events_tt_train[(dl_mask) & (mask_two_tauhad)]# checked


for column, label, label2, func, borders in zip([[events_sl_notau.run3_dnn_moe_hh, events_sl_1tau.run3_dnn_moe_hh, events_dl_onetau.run3_dnn_moe_hh, events_fh.run3_dnn_moe_hh, events_dl_tautau.run3_dnn_moe_hh, events_tt_train.run3_dnn_moe_hh, events_hh.run3_dnn_moe_hh, events_kl1.run3_dnn_moe_hh, events_kl0.run3_dnn_moe_hh, events_kl245.run3_dnn_moe_hh ,events_kl5.run3_dnn_moe_hh],
                                        [events_sl_notau.ll_mass, events_sl_1tau.ll_mass, events_dl_onetau.ll_mass, events_fh.ll_mass, events_dl_tautau.ll_mass, events_tt_train.ll_mass, events_hh.ll_mass, events_kl1.ll_mass, events_kl0.ll_mass, events_kl245.ll_mass, events_kl5.ll_mass],
                                        [events_sl_notau.met_pt, events_sl_1tau.met_pt, events_dl_onetau.met_pt, events_fh.met_pt, events_dl_tautau.met_pt, events_tt_train.met_pt, events_hh.met_pt, events_kl1.met_pt, events_kl0.met_pt, events_kl245.met_pt, events_kl5.met_pt],
                                        [events_sl_notau.bb_mass, events_sl_1tau.bb_mass, events_dl_onetau.bb_mass, events_fh.bb_mass, events_dl_tautau.bb_mass, events_tt_train.bb_mass, events_hh.bb_mass, events_kl1.bb_mass, events_kl0.bb_mass, events_kl245.bb_mass, events_kl5.bb_mass],
                                        [events_sl_notau.llbb_mass, events_sl_1tau.llbb_mass, events_dl_onetau.llbb_mass, events_fh.llbb_mass, events_dl_tautau.llbb_mass, events_tt_train.llbb_mass, events_hh.llbb_mass, events_kl1.llbb_mass, events_kl0.llbb_mass, events_kl245.llbb_mass, events_kl5.llbb_mass]],
                                        ["HH output node", "lepton mass", r"MET $p_T$", "bb mass", "llbb mass"],
                                        ["HHdnn", "m_ll", "MET", "m_bb", "m_llbb"],
                                        [logit, identity, identity, identity, identity],
                                        [[-14, 8],[0,150], [-10, 500], [0, 300], [0, 800]]):
    sl_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    sltau_hist        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    dl_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    fh_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    tautau_hist       = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    all_hist          = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_hist       = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl1        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl0        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl245      = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl5        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))

    sl_hist.fill(func(column[0]), weight =events_sl_notau.event_weight)
    sltau_hist.fill(func(column[1]), weight =events_sl_1tau.event_weight)
    dl_hist.fill(func(column[2]), weight =events_dl_onetau.event_weight)
    fh_hist.fill(func(column[3]), weight =events_fh.event_weight)
    tautau_hist.fill(func(column[4]), weight =events_dl_tautau.event_weight)
    all_hist.fill(func(column[5]), weight =events_tt_train.event_weight)
    signal_hist.fill(func(column[6]), weight =events_hh.event_weight)
    signal_histkl1.fill(func(column[7]), weight =events_kl1.event_weight)
    signal_histkl0.fill(func(column[8]), weight =events_kl0.event_weight)
    signal_histkl245.fill(func(column[9]), weight =events_kl245.event_weight)
    signal_histkl5.fill(func(column[10]), weight =events_kl5.event_weight)

    # scale the hh histogram up, weighted by the integral of the tt data
    scaling_factor = ((signal_histkl1.values().sum())/ (all_hist.values().sum()))**(-1)

    # plot
    x = np.linspace(borders[0], borders[1], n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Number of events', color="black")
    ax1.step(x, sl_hist.values(), alpha=0.9, label=r"tt: sl decay, no $\tau_h$", color='green')
    ax1.step(x, sltau_hist.values(), alpha=0.9, label=r"tt: sl decay, one $\tau_h$", color='tab:brown')
    ax1.step(x, dl_hist.values(), alpha=0.9, label=r"tt: dl decay, one $\tau_h$", color='blue')
    ax1.step(x, tautau_hist.values(), alpha=0.9, label=r"tt: dl decay, two $\tau_h$", color='tab:orange')
    ax1.step(x, fh_hist.values(), alpha=0.9, label=r"tt: fh decay", color='tab:pink')
    ax1.step(x, all_hist.values(), alpha=0.9, label=r"tt: all events", color='red')
    # ax1.step(x, signal_hist.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 0, 1, 2.45)x {round(scaling_factor)}", color='black')
    # ax1.step(x, signal_histkl0.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 0) x {round(scaling_factor)}", color="darkslategray")
    ax1.step(x, signal_histkl1.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 1) x {round(scaling_factor)}", color="black")
    # ax1.step(x, signal_histkl245.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 2.45) x {round(scaling_factor)}", color="silver")
    # ax1.step(x, signal_histkl5.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 5) x {round(scaling_factor)}", color="gray")


    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_legend_handles_labels()
    plt.legend(fontsize="small")
    ax1.set_yscale("log")
    ax1.set_xscale("linear")
    ax1.set_ylim(bottom=1e-1)
    fig.tight_layout()
    plt.title(fr"{label}; signal and tt background events, etau, res2b category, split in W decay modes; only hadronically decaying taus are considered taus", wrap=True)
    plt.savefig(f"analysis_etau/{label2}_etaures2b_W_decay_mode", dpi=300, bbox_inches='tight')
    plt.show()

    sl_hist.reset()
    dl_hist.reset()
    fh_hist.reset()
    tautau_hist.reset()
    all_hist.reset()
    signal_histkl1.reset()
    signal_histkl0.reset()
    signal_histkl245.reset()
    signal_histkl5.reset()

# hadronic tau matching
tau_matched_unknown = events_tt[ak.flatten(events_tt.tau_genPartFlav == 0)]
tau_matched_to_e   = events_tt[ak.flatten(events_tt.tau_genPartFlav == 1)]
tau_matched_to_mu  = events_tt[ak.flatten(events_tt.tau_genPartFlav == 2)]
tau_matched_to_tau_emu = ak.concatenate([events_tt[ak.flatten(events_tt.tau_genPartFlav == 3)], events_tt[ak.flatten(events_tt.tau_genPartFlav == 4)]])
tau_matched_to_tau_h = events_tt[ak.flatten(events_tt.tau_genPartFlav == 5)]

for column, label, label2, func, borders in zip([[tau_matched_to_e.run3_dnn_moe_hh, tau_matched_to_mu.run3_dnn_moe_hh, tau_matched_to_tau_h.run3_dnn_moe_hh, tau_matched_to_tau_emu.run3_dnn_moe_hh, tau_matched_unknown.run3_dnn_moe_hh, events_tt.run3_dnn_moe_hh, events_hh.run3_dnn_moe_hh, events_kl1.run3_dnn_moe_hh, events_kl0.run3_dnn_moe_hh, events_kl245.run3_dnn_moe_hh, events_kl5.run3_dnn_moe_hh],
                                        [tau_matched_to_e.ll_mass, tau_matched_to_mu.ll_mass, tau_matched_to_tau_h.ll_mass, tau_matched_to_tau_emu.ll_mass, tau_matched_unknown.ll_mass, events_tt.ll_mass, events_hh.ll_mass, events_kl1.ll_mass, events_kl0.ll_mass, events_kl245.ll_mass, events_kl5.ll_mass],
                                        [tau_matched_to_e.met_pt, tau_matched_to_mu.met_pt, tau_matched_to_tau_h.met_pt, tau_matched_to_tau_emu.met_pt, tau_matched_unknown.met_pt, events_tt.met_pt, events_hh.met_pt, events_kl1.met_pt, events_kl0.met_pt, events_kl245.met_pt, events_kl5.met_pt]],
                                        ["HH output node", "lepton mass", r"MET $p_T$"],#, "WW mass"],
                                        ["HHdnn", "m_ll", "MET"],#, "m_WW"],
                                        [logit, identity, identity],#, identity],
                                        [[-14, 8],[0,150], [-10, 500]]):#,[0,160]]):
    matched2e_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2mu_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2tauhad_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2tauemu_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    matched2unknown_hist       = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    all_hist          = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl1        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl0        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl245      = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl5        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))

    matched2e_hist.fill(func(column[0]), weight =tau_matched_to_e.event_weight)
    matched2mu_hist.fill(func(column[1]), weight =tau_matched_to_mu.event_weight)
    matched2tauhad_hist.fill(func(column[2]), weight =tau_matched_to_tau_h.event_weight)
    matched2tauemu_hist.fill(func(column[3]), weight =tau_matched_to_tau_emu.event_weight)
    matched2unknown_hist.fill(func(column[4]), weight =tau_matched_unknown.event_weight)
    all_hist.fill(func(column[5]), weight =events_tt_train.event_weight)
    signal_hist.fill(func(column[6]), weight =events_hh.event_weight)
    signal_histkl1.fill(func(column[7]), weight =events_kl1.event_weight)
    signal_histkl0.fill(func(column[8]), weight =events_kl0.event_weight)
    signal_histkl245.fill(func(column[9]), weight =events_kl245.event_weight)
    signal_histkl5.fill(func(column[10]), weight =events_kl5.event_weight)

    # scale the hh histogram up, weighted by the integral of the tt data
    scaling_factor = ((signal_histkl1.values().sum())/ (all_hist.values().sum()))**(-1)

    # plot
    x = np.linspace(borders[0], borders[1], n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Number of events', color="black")
    ax1.step(x, matched2e_hist.values(), alpha=0.9, label=r"tt: reco $\tau$ matched to e", color='green')
    ax1.step(x, matched2mu_hist.values(), alpha=0.9, label=r"tt: reco $\tau$ matched to $\mu$", color='blue')
    ax1.step(x, matched2tauhad_hist.values(), alpha=0.9, label=r"tt: reco $\tau$ matched to $\tau_h$", color='tab:orange')
    ax1.step(x, matched2tauemu_hist.values(), alpha=0.9, label=r"tt: reco $\tau$ matched to $\tau_e$, $\tau_\mu$", color='tab:brown')
    ax1.step(x, matched2unknown_hist.values(), alpha=0.9, label=r"tt: unknown origin of reco $\tau$", color='tab:pink')
    ax1.step(x, all_hist.values(), alpha=0.9, label=r"tt: all events", color='red')
    # ax1.step(x, signal_hist.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 0, 1, 2.45)x {round(scaling_factor)}", color='black')
    # ax1.step(x, signal_histkl0.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 0) x {round(scaling_factor)}", color="darkslategray")
    ax1.step(x, signal_histkl1.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 1) x {round(scaling_factor)}", color="black")
    # ax1.step(x, signal_histkl245.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 2.45) x {round(scaling_factor)}", color="silver")
    # ax1.step(x, signal_histkl5.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 5) x {round(scaling_factor)}", color="gray")


    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_legend_handles_labels()
    plt.legend(fontsize="small")
    ax1.set_yscale("log")
    ax1.set_xscale("linear")
    ax1.set_ylim(bottom=1e-1)
    fig.tight_layout()
    plt.title(fr"{label}; $\tau$ matching for signal and tt background events, etau, res2b category", wrap=True)
    plt.savefig(f"analysis_etau/{label2}_etaures2b_tau_matching", dpi=300, bbox_inches='tight')
    plt.show()

    matched2e_hist.reset()
    matched2mu_hist.reset()
    matched2tauhad_hist.reset()
    matched2tauemu_hist.reset()
    matched2unknown_hist.reset()
    all_hist.reset()
    signal_histkl1.reset()
    signal_histkl0.reset()
    signal_histkl245.reset()
    signal_histkl5.reset()

##################################################################################################################
##################################################################################################################
##################################################################################################################
# b matching
# delr of first b
delr1_b = deltaR(
    events_tt_train.bjet_eta[:, 0],
    events_tt_train.bjet_phi[:, 0],
    events_tt_train.gen_top_b_eta[:, 0],
    events_tt_train.gen_top_b_phi[:, 0],
)
delr2_b = deltaR(
    events_tt_train.bjet_eta[:, 0],
    events_tt_train.bjet_phi[:, 0],
    events_tt_train.gen_top_b_eta[:, 1],
    events_tt_train.gen_top_b_phi[:, 1],
)
# delr of 2nd b
delr3_b = deltaR(
    events_tt_train.bjet_eta[:, 1],
    events_tt_train.bjet_phi[:, 1],
    events_tt_train.gen_top_b_eta[:, 0],
    events_tt_train.gen_top_b_phi[:, 0]
)
delr4_b = deltaR(
    events_tt_train.bjet_eta[:, 1],
    events_tt_train.bjet_phi[:, 1],
    events_tt_train.gen_top_b_eta[:, 1],
    events_tt_train.gen_top_b_phi[:, 1],
)

min_delr_emu1 = np.minimum(delr1_emu, delr2_emu) # first W
min_delr_emu2 = np.minimum(delr3_emu, delr4_emu) # second W
# merge to one array for min delta r of both bjets
delta_rs = np.stack([min_delr_emu1, min_delr_emu2], axis=1)
delta_rs = ak.Array(delta_rs)

# b matching
two_b_matched    = events_tt_train[(np.minimum(delr1_b, delr2_b) < delr_cut_b) & (np.minimum(delr3_b, delr4_b) < delr_cut_b)]
onebmatched_1bfake = events_tt_train[(np.minimum(delr1_b, delr2_b) < delr_cut_b) & (~(np.minimum(delr3_b, delr4_b) < delr_cut_b)) |
                                   (~(np.minimum(delr1_b, delr2_b) < delr_cut_b)) & (np.minimum(delr3_b, delr4_b) < delr_cut_b)]
twobfake           = events_tt_train[(~(np.minimum(delr1_b, delr2_b) < delr_cut_b)) & (~(np.minimum(delr3_b, delr4_b) < delr_cut_b))]

print("done with b matching")

# func = logit
# # initialize hists
# # fh match hist not necessary, as all fh events are unmatched (which we expected)
# two_b = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
# oneb_onefake = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
# twofake = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
# allevents = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))

# # fill hists
# two_b.fill(func(two_b_matched.run3_dnn_moe_hh), weight =two_b_matched.event_weight)
# oneb_onefake.fill(func(onebmatched_1bfake.run3_dnn_moe_hh), weight =onebmatched_1bfake.event_weight)
# twofake.fill(func(twobfake.run3_dnn_moe_hh), weight =twobfake.event_weight)
# allevents.fill(func(events_tt_train.run3_dnn_moe_hh), weight =events_tt_train.event_weight)

# # plot histograms
# x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
# x = (x[:-1] + x[1:]) / 2  # bin centers

# fig, ax1 = plt.subplots(figsize=(9, 5))
# fig.subplots_adjust(right=0.85)
# ax1.set_xlabel('HH output node')
# ax1.set_ylabel('Number of events', color="black")
# ax1.step(x, list(two_b.values()), alpha=0.9, label=r'zero fake b jets', color='green')
# ax1.step(x, list(oneb_onefake.values()), alpha=0.9, label=r'one fake b jet', color='limegreen')
# ax1.step(x, list(twofake.values()), alpha=0.9, label=r'two fake b jets', color='purple')
# ax1.step(x, list(allevents.values()), alpha=0.9, label=r'all events', color='red')


# ax1.tick_params(axis='y', labelcolor="black")
# ax1.get_legend_handles_labels()
# plt.legend()
# ax1.set_yscale("log")
# ax1.set_xscale("linear")
# ax1.set_ylim(bottom=1e-1)
# ax1.legend(loc='upper right')
# fig.tight_layout()
# plt.title(r"HH output node; tt bg of etau, res2b category split in correctly matched and fake b events (matching criterion: $\Delta R <$"+f" {delr_cut_b})", wrap=True)
# plt.savefig("analysis_etau/res2b_dnn_b_matching", dpi=300, bbox_inches='tight')
# plt.show()

for column, label, label2, func, borders in zip([[two_b_matched.run3_dnn_moe_hh, onebmatched_1bfake.run3_dnn_moe_hh, twobfake.run3_dnn_moe_hh, events_tt_train.run3_dnn_moe_hh, events_hh.run3_dnn_moe_hh, events_kl1.run3_dnn_moe_hh, events_kl0.run3_dnn_moe_hh, events_kl245.run3_dnn_moe_hh, events_kl5.run3_dnn_moe_hh],
                                        [two_b_matched.bb_mass, onebmatched_1bfake.bb_mass, twobfake.bb_mass, events_tt_train.bb_mass, events_hh.bb_mass, events_kl1.bb_mass,events_kl0.bb_mass, events_kl245.bb_mass, events_kl5.bb_mass]],
                                        ["HH output node", "bb mass"],
                                        ["HHdnn","m_bb"],
                                        [logit, identity],
                                        [[-14, 8], [0, 300]]):
    two_b_hist            = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    one_b_hist            = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    zero_b_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    all_hist              = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_his            = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl1        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl0        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl245      = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    signal_histkl5        = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    two_b_hist.fill(func(column[0]), weight =two_b_matched.event_weight)
    one_b_hist.fill(func(column[1]), weight =onebmatched_1bfake.event_weight)
    zero_b_hist.fill(func(column[2]), weight =twobfake.event_weight)
    all_hist.fill(func(column[3]), weight =events_tt_train.event_weight)
    signal_hist.fill(func(column[4]), weight =events_hh.event_weight)
    signal_histkl1.fill(func(column[5]), weight =events_kl1.event_weight)
    signal_histkl0.fill(func(column[6]), weight =events_kl0.event_weight)
    signal_histkl245.fill(func(column[7]), weight =events_kl245.event_weight)
    signal_histkl5.fill(func(column[8]), weight =events_kl5.event_weight)

    # scale the hh histogram up, weighted by the integral of the tt data
    scaling_factor = ((signal_histkl1.values().sum())/ (all_hist.values().sum()))**(-1)# + signal_histkl5.values().sum()

    # plot
    x = np.linspace(borders[0], borders[1], n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Number of events', color="black")
    ax1.step(x, two_b_hist.values(), alpha=0.9, label="tt: Zero fake b jets", color='green')
    ax1.step(x, one_b_hist.values(), alpha=0.9, label="tt: One fake b jet", color='blue')
    ax1.step(x, zero_b_hist.values(), alpha=0.9, label="tt: Two fake b jets", color='tab:pink')
    ax1.step(x, all_hist.values(), alpha=0.9, label="tt: all events", color='red')
    # ax1.step(x, signal_hist.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 0, 1, 2.45)x {round(scaling_factor)}", color='black')
    # ax1.step(x, signal_histkl0.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 0) x {round(scaling_factor)}", color="darkslategray")
    ax1.step(x, signal_histkl1.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 1) x {round(scaling_factor)}", color="black")
    # ax1.step(x, signal_histkl245.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 2.45) x {round(scaling_factor)}", color="silver")
    # ax1.step(x, signal_histkl5.values()* scaling_factor, alpha=0.9, label=fr"signal ($\kappa_\lambda$ = 5) x {round(scaling_factor)}", color="gray")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_legend_handles_labels()
    plt.legend(fontsize="small")
    ax1.set_yscale("log")
    ax1.set_xscale("linear")
    ax1.set_ylim(bottom=1e-1)
    fig.tight_layout()
    plt.title(fr"{label}; signal and tt background events, etau, res2b category, split in number of b fakes", wrap=True)
    plt.savefig(f"analysis_etau/bmatching_{label2}_etaures2b", dpi=300, bbox_inches='tight')
    plt.show()

    two_b_hist.reset()
    one_b_hist.reset()
    zero_b_hist.reset()
    all_hist.reset()
    signal_histkl1.reset()
    signal_histkl0.reset()
    signal_histkl245.reset()
    signal_histkl5.reset()
