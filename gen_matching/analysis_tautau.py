import itertools

import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt


"""This script analyses the tautau channel and matches tau to gen W children.
We separate between fully hadronic, semi-leptonic and di-leptonic channels.
For the different cases, we identify if the taus are fakes or not, with the aim of
analysing the different cases concerning their HH DNN output score distribution.
"""
n_bins = 100

eps = 1e-6 # set eps=0 for normal scale
lower_border = -14# set to 0 for lin scale
upper_border = 9# set to 1 for lin scale
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
# events_tt = events_tt[events_tt.channel_id == 3] # tautau channel
events_tt = events_tt[ak.any(events_tt.category_ids == 207, axis = 1)] # tautau, res2b channel
events_tt_train = events_tt
# events_tt_train = ak.concatenate([events_tt[:10000], events_tt[844445:854446]]) # first ev are dl, second sl
# events_tt_train = ak.concatenate([events_tt_train, events_tt[844127:844444,]]) # also add fh events

# maybe double-check if still only dl events pass the tautau channel cuts
events_sl = events_tt_train[events_tt_train.process_id == 1100]
events_dl = events_tt_train[events_tt_train.process_id == 1200]
events_fh = events_tt_train[events_tt_train.process_id == 1300]

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

##########################################################################################################
# # plot all events
# tautau_fh = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="fully hadronic", label=""))
# tautau_sl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="semi-leptonic", label=""))
# tautau_dl = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="di-leptonic", label=""))
# all_ev = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="all events", label=""))

# # fill hists
# tautau_fh.fill(func(events_fh.run3_dnn_moe_hh), weight =events_fh.event_weight)
# tautau_sl.fill(func(events_sl.run3_dnn_moe_hh), weight =events_sl.event_weight)
# tautau_dl.fill(func(events_dl.run3_dnn_moe_hh), weight =events_dl.event_weight)
# all_ev.fill(func(events_tt_train.run3_dnn_moe_hh), weight =events_tt_train.event_weight)

# x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
# x = (x[:-1] + x[1:]) / 2  # bin centers
# fig, ax1 = plt.subplots(figsize=(9, 5))
# fig.subplots_adjust(right=0.85)
# ax1.set_xlabel('HH output node')
# ax1.set_ylabel('Number of events', color="black")
# ax1.step(x, tautau_fh.values(), alpha=0.9, label="fully hadronic", color='green')
# ax1.step(x, tautau_sl.values(), alpha=0.9, label="semi-leptonic", color='tab:orange')
# ax1.step(x, tautau_dl.values(), alpha=0.9, label="di-leptonic", color='tab:purple')
# ax1.step(x, all_ev.values(), alpha=0.9, label="all events", color='red')
# plt.legend()
# ax1.set_yscale("log")
# ax1.set_xscale("linear")
# ax1.set_ylim(bottom=1e-1)
# fig.tight_layout()
# plt.title(fr"HH output node; tt background events, tautau channel", wrap=True)
# plt.savefig("analysis_tautau/dnn_allevents", dpi=300, bbox_inches='tight')
# plt.show()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# plot sl, dl, fh together
# W decay mode
sl_mask = events_tt_train.process_id == 1100
dl_mask = events_tt_train.process_id == 1200
fh_mask = events_tt_train.process_id == 1300
mask_two_taus   = ak.all(ak.any(abs(events_tt_train.gen_top_w_children_pdgId) == 15, axis=2), axis=1)
events_sl = events_tt_train[sl_mask]
events_dl = events_tt_train[(dl_mask) & (~mask_two_taus)]
events_fh = events_tt_train[fh_mask]
events_tautau = events_tt_train[(dl_mask) & (mask_two_taus)]

for column, label, label2, func, borders in zip([[events_sl.run3_dnn_moe_hh, events_dl.run3_dnn_moe_hh, events_fh.run3_dnn_moe_hh, events_tautau.run3_dnn_moe_hh, events_tt_train.run3_dnn_moe_hh],
                                        [events_sl.ll_mass, events_dl.ll_mass, events_fh.ll_mass, events_tautau.ll_mass, events_tt_train.ll_mass]],#[ak.flatten(events_sl.gen_top_w_mass), ak.flatten(events_dl.gen_top_w_mass), ak.flatten(events_fh.gen_top_w_mass), ak.flatten(events_tautau.gen_top_w_mass), ak.flatten(events_tt_train.gen_top_w_mass)]],
                                        ["HH output node", "lepton mass"],#, "WW mass"],
                                        ["HHdnn", "m_ll"],#, "m_WW"],
                                        [logit, identity],#, identity],
                                        [[-14, 8],[0,150]]):#,[0,160]]):
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
    plt.title(fr"{label}; tt background events, etau, res2b category, split in W decay modes", wrap=True)
    plt.savefig(f"analysis_tautau/{label2}_etaures2b_W_decay_mode", dpi=300, bbox_inches='tight')
    plt.show()

    sl_hist.reset()
    dl_hist.reset()
    fh_hist.reset()
    tautau_hist.reset()
    all_hist.reset()

# hadronic tau matching
twomatchedtauhad = events_tt[ak.all(events_tt.tau_genPartFlav == 5, axis=-1)]
onematchedemu_onematchedtauh = events_tt[ak.any(events_tt.tau_genPartFlav == 3, axis=-1) & ak.any(events_tt.tau_genPartFlav == 5, axis=-1)]
twofakes_unknown = events_tt[ak.all(events_tt.tau_genPartFlav == 0, axis=-1)]
onefake = events_tt[(ak.any(events_tt.tau_genPartFlav == 3, axis=-1) | ak.any(events_tt.tau_genPartFlav == 4, axis=-1) | ak.any(events_tt.tau_genPartFlav == 5, axis=-1)) &
                     (ak.any(events_tt.tau_genPartFlav == 0, axis=-1) | ak.any(events_tt.tau_genPartFlav == 1, axis=-1) | ak.any(events_tt.tau_genPartFlav == 2, axis=-1))]
twofakes_others = events_tt[(~ak.all(events_tt.tau_genPartFlav == 5, axis=-1)) &
                            (~(ak.any(events_tt.tau_genPartFlav == 3, axis=-1) & ak.any(events_tt.tau_genPartFlav == 5, axis=-1))) &
                            (~ak.all(events_tt.tau_genPartFlav == 0, axis=-1)) &
                            (~((ak.any(events_tt.tau_genPartFlav == 3, axis=-1) | ak.any(events_tt.tau_genPartFlav == 4, axis=-1) | ak.any(events_tt.tau_genPartFlav == 5, axis=-1)) &
                               (ak.any(events_tt.tau_genPartFlav == 0, axis=-1) | ak.any(events_tt.tau_genPartFlav == 1, axis=-1) | ak.any(events_tt.tau_genPartFlav == 2, axis=-1))))]

for column, label, label2, func, borders in zip([[twomatchedtauhad.run3_dnn_moe_hh, onematchedemu_onematchedtauh.run3_dnn_moe_hh, onefake.run3_dnn_moe_hh, twofakes_unknown.run3_dnn_moe_hh, twofakes_others.run3_dnn_moe_hh,  events_tt.run3_dnn_moe_hh],
                                        [twomatchedtauhad.ll_mass, onematchedemu_onematchedtauh.ll_mass, onefake.ll_mass, twofakes_unknown.ll_mass, twofakes_others.ll_mass, events_tt.ll_mass]],#[ak.flatten(twomatchedtauhad.gen_top_w_mass), ak.flatten(onematchedemu_onematchedtauh.gen_top_w_mass), ak.flatten(twofakes_unknown.gen_top_w_mass), ak.flatten(twofakes_others.gen_top_w_mass), ak.flatten(events_tt_train.gen_top_w_mass)]],
                                        ["HH output node", "lepton mass"],#, "WW mass"],
                                        ["HHdnn", "m_ll"],#, "m_WW"],
                                        [logit, identity],#, identity],
                                        [[-14, 8],[0,150]]):#,[0,160]]):
    twomatchedtauhad_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    onematchedemu_onematchedtauh_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    onefake_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    twofakes_unknown_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    twofakes_others_hist           = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))
    all_hist          = Hist(hist.axis.Regular(n_bins, borders[0], borders[1], name="", label=r""))

    twomatchedtauhad_hist.fill(func(column[0]), weight =twomatchedtauhad.event_weight)
    onematchedemu_onematchedtauh_hist.fill(func(column[1]), weight =onematchedemu_onematchedtauh.event_weight)
    onefake_hist.fill(func(column[2]), weight =onefake.event_weight)
    twofakes_unknown_hist.fill(func(column[3]), weight =twofakes_unknown.event_weight)
    twofakes_others_hist.fill(func(column[4]), weight =twofakes_others.event_weight)
    all_hist.fill(func(column[5]), weight =events_tt_train.event_weight)
    # plot
    x = np.linspace(borders[0], borders[1], n_bins + 1)  # bin edges
    x = (x[:-1] + x[1:]) / 2  # bin centers
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.85)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Number of events', color="black")
    ax1.step(x, twomatchedtauhad_hist.values(), alpha=0.9, label=r"2 reco $\tau$ matched to $\tau_h$", color='green')
    ax1.step(x, onematchedemu_onematchedtauh_hist.values(), alpha=0.9, label=rf"1 reco $\tau$ matched to $\tau_h$, 1 to $\tau_e$ or $\tau_\mu$", color='blue')
    ax1.step(x, onefake_hist.values(), alpha=0.9, label=rf"1 reco $\tau$ is fake", color='tab:pink')
    ax1.step(x, twofakes_unknown_hist.values(), alpha=0.9, label=r"2 reco $\tau$ with unknown origin", color='tab:orange')
    ax1.step(x, twofakes_others_hist.values(), alpha=0.9, label=r"Other events with 2 $\tau$ fakes", color='tab:brown')
    ax1.step(x, all_hist.values(), alpha=0.9, label=r"all events", color='red')

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_legend_handles_labels()
    plt.legend()
    ax1.set_yscale("log")
    ax1.set_xscale("linear")
    ax1.set_ylim(bottom=1e-1)
    fig.tight_layout()
    plt.title(fr"{label}; $\tau$ matching for tt background events, tautau, res2b category", wrap=True)
    plt.savefig(f"analysis_tautau/{label2}_tautaures2b_tau_matching", dpi=300, bbox_inches='tight')
    plt.show()

    twomatchedtauhad_hist.reset()
    onematchedemu_onematchedtauh_hist.reset()
    onefake_hist.reset()
    twofakes_unknown_hist.reset()
    twofakes_others_hist.reset()
    all_hist.reset()
from IPython import embed; embed(header="MESSAGE Line 212 | File: analysis_tautau.py")

##########################################################################################################
##########################################################################################################
# dl decay, tau gen matching
delr1_tau = deltaR(
    events_dl.tau_eta[:],
    events_dl.tau_phi[:],
    events_dl.gen_top_w_children_eta[:, 0, 0],
    events_dl.gen_top_w_children_phi[:, 0, 0],
)
delr2_tau = deltaR(
    events_dl.tau_eta[:],
    events_dl.tau_phi[:],
    events_dl.gen_top_w_children_eta[:, 0, 1],
    events_dl.gen_top_w_children_phi[:, 0, 1],
)
delr3_tau = deltaR(
    events_dl.tau_eta[:],
    events_dl.tau_phi[:],
    events_dl.gen_top_w_children_eta[:, 1, 0],
    events_dl.gen_top_w_children_phi[:, 1, 0],
)
delr4_tau = deltaR(
    events_dl.tau_eta[:],
    events_dl.tau_phi[:],
    events_dl.gen_top_w_children_eta[:, 1, 1],
    events_dl.gen_top_w_children_phi[:, 1, 1],
)
# tau_delrs = np.zeros((len(events_dl), 2))
# tau_matches = np.zeros((len(events_dl), 2, 2)) # which children
# nb_of_taus_dl = np.ones(len(events_dl), dtype=np.int_)
# multiple_matches_indices = []

# for i in range(len(events_dl)):
#     for j in range(len(delr1_tau[i])):
#         if delr1_tau[i][j] < delr2_tau[i][j]:
#             tau_matches[i][j][0] = 1
#             tau_delrs[i][j] = delr1_tau[i][j]
#         else:
#             tau_matches[i][j][1] = 1
#             tau_delrs[i][j] = delr1_tau[i][j]
#     for k in range(len(delr1_tau[i])):
#         if delr3_tau[i][k] < delr4_tau[i][k]:
#             tau_matches[i][k][0] = 1
#             tau_delrs[i][k] = delr3_tau[i][k]
#         else:
#             tau_matches[i][k][1] = 1
#             tau_delrs[i][k] = delr4_tau[i][k]
#     nb_of_taus_dl[i] = len(delr1_tau)

#
delta_w1_close_first_child_mask = (delr1_tau < delr2_tau) # shape [(tau1, w1child1), (tau2, w1child1)]
delta_w1_close_second_child_mask = ~delta_w1_close_first_child_mask # shape [(tau1, w1child2), (tau2, w1child2)]
delta_w2_close_first_child_mask = (delr3_tau < delr4_tau)
delta_w2_close_second_child_mask = ~delta_w2_close_first_child_mask

tau_matches_w1 = ak.concatenate((delta_w1_close_first_child_mask, delta_w1_close_second_child_mask), axis=-1)
tau_matches_w2 = ak.concatenate((delta_w2_close_first_child_mask, delta_w2_close_second_child_mask), axis=-1)
tau_matches_w1 = ak.unflatten(tau_matches_w1, 2, axis=1) # shape [[(tau1, w1child1), (tau2, w1child1)],[(tau1, w1child2), (tau2, w1child2)]]
tau_matches_w2 = ak.unflatten(tau_matches_w2, 2, axis=1) # shape [[(tau1, w2child1), (tau2, w2child1)],[(tau1, w2child2), (tau2, w2child2)]]
def is_t_from_multiple_W(ak_array):
    # check this when working with full dataset
    # expects array in form of [1st Child 1st W, 2nd Child 1st W],[1 Child 2nd W, ...]
    first_tau = ak.sum(ak.sum(a[:, :, 0], axis=1) > 1)
    second_tau = ak.sum(ak.sum(a[:, :, 1], axis=1) > 1)
    check = (first_tau > 0 or second_tau > 0)
    if any(check):
        print(f"First or Second Tau originates from multiple W: {check}")
        return False
    return True

# re-build arrays in such a way that shape is [[w1 child1, w1 child2],[w2 child 1, w1 child2]] for each tau
tau1_w1_c1_mask = delta_w1_close_first_child_mask[:, 0]
tau1_w1_c2_mask = delta_w1_close_second_child_mask[:, 0]
tau1_w2_c1_mask = delta_w2_close_first_child_mask[:, 0]
tau1_w2_c2_mask = delta_w2_close_second_child_mask[:, 0]
tau1_w1_mask = ak.concatenate((ak.unflatten(tau1_w1_c1_mask, 1, axis=0), ak.unflatten(tau1_w1_c2_mask, 1, axis=0)), axis=-1)
tau1_w2_mask = ak.concatenate((ak.unflatten(tau1_w2_c1_mask, 1, axis=0), ak.unflatten(tau1_w2_c2_mask, 1, axis=0)), axis=-1)
tau1_mask = ak.unflatten(ak.concatenate((tau1_w1_mask, tau1_w2_mask), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]

tau2_w1_c1_mask = delta_w1_close_first_child_mask[:, 1]
tau2_w1_c2_mask = delta_w1_close_second_child_mask[:, 1]
tau2_w2_c1_mask = delta_w2_close_first_child_mask[:, 1]
tau2_w2_c2_mask = delta_w2_close_second_child_mask[:, 1]
tau2_w1_mask = ak.concatenate((ak.unflatten(tau2_w1_c1_mask, 1, axis=0), ak.unflatten(tau2_w1_c2_mask, 1, axis=0)), axis=-1)
tau2_w2_mask = ak.concatenate((ak.unflatten(tau2_w2_c1_mask, 1, axis=0), ak.unflatten(tau2_w2_c2_mask, 1, axis=0)), axis=-1)
tau2_mask = ak.unflatten(ak.concatenate((tau2_w1_mask, tau2_w2_mask), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]
# do the same for delr
delr_tau1_w1_c1 = delr1_tau[:, 0]
delr_tau1_w1_c2 = delr2_tau[:, 0]
delr_tau1_w2_c1 = delr3_tau[:, 0]
delr_tau1_w2_c2 = delr4_tau[:, 0]
delr_tau1_w1 = ak.concatenate((ak.unflatten(delr_tau1_w1_c1, 1, axis=0), ak.unflatten(delr_tau1_w1_c2, 1, axis=0)), axis=-1)
delr_tau1_w2 = ak.concatenate((ak.unflatten(delr_tau1_w2_c1, 1, axis=0), ak.unflatten(delr_tau1_w2_c2, 1, axis=0)), axis=-1)
delr_tau1 = ak.unflatten(ak.concatenate((delr_tau1_w1, delr_tau1_w2), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]

delr_tau2_w1_c1 = delr1_tau[:, 1]
delr_tau2_w1_c2 = delr2_tau[:, 1]
delr_tau2_w2_c1 = delr3_tau[:, 1]
delr_tau2_w2_c2 = delr4_tau[:, 1]
delr_tau2_w1 = ak.concatenate((ak.unflatten(delr_tau2_w1_c1, 1, axis=0), ak.unflatten(delr_tau2_w1_c2, 1, axis=0)), axis=-1)
delr_tau2_w2 = ak.concatenate((ak.unflatten(delr_tau2_w2_c1, 1, axis=0), ak.unflatten(delr_tau2_w2_c2, 1, axis=0)), axis=-1)
delr_tau2 = ak.unflatten(ak.concatenate((delr_tau2_w1, delr_tau2_w2), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]
mask_match_tau1 = ak.any(ak.any(delr_tau1 < delr_cut_tau, axis=-1), axis=-1)
mask_match_tau2 = ak.any(ak.any(delr_tau2 < delr_cut_tau, axis=-1), axis=-1)

matched_delr_tau1 = delr_tau1[mask_match_tau1]
matched_delr_tau2 = delr_tau2[mask_match_tau2]
#
mask_tau_event = ak.any(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 15, axis=2), axis=1)
mask_e_event   = ak.any(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 11, axis=2), axis=1)
mask_mu_event  = ak.any(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 13, axis=2), axis=1)
mask_two_taus   = ak.all(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 15, axis=2), axis=1)
mask_two_es   = ak.all(ak.any(abs(events_dl.gen_top_w_children_pdgId) == 11, axis=2), axis=1)


events_2matched_2tau     = events_dl[mask_match_tau1 & mask_match_tau2 & mask_two_taus]
events_2matched_1tau_1e  = events_dl[mask_match_tau1 & mask_match_tau2 & mask_e_event & mask_tau_event]
events_2matched_1tau_1mu = events_dl[mask_match_tau1 & mask_match_tau2 & mask_mu_event & mask_tau_event]
events_2matched_2e       = events_dl[mask_match_tau1 & mask_match_tau2 & mask_two_es]

events_1matchedtau = events_dl[ mask_match_tau1 & ~mask_match_tau2 & ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 15, axis=1)|# first tau candidate is match and real tau
                               ~mask_match_tau1 & mask_match_tau2   &ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 15, axis=1)]# 2nd tau candidate is match and real tau
events_1match_e_fake = events_dl[ mask_match_tau1 & ~mask_match_tau2 & ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 11, axis=1)|# first tau candidate is match and is e
                               ~mask_match_tau1 & mask_match_tau2   &ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 11, axis=1)]# 2nd tau candidate is match and is e
events_1match_mu_fake = events_dl[ mask_match_tau1 & ~mask_match_tau2 & ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 13, axis=1)|# first tau candidate is match and is e
                               ~mask_match_tau1 & mask_match_tau2   &ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 13, axis=1)]# 2nd tau candidate is match and is e
# events_no_matches = events_dl[~mask_match_tau1 & ~mask_match_tau2] # dont exist

delr_1match      = delr_tau1[ mask_match_tau1 & ~mask_match_tau2 |
                             ~mask_match_tau1 &  mask_match_tau2]
delr2_1match      = delr_tau2[ mask_match_tau1 & ~mask_match_tau2 |
                             ~mask_match_tau1 &  mask_match_tau2]
events_1match   = events_dl[ mask_match_tau1 & ~mask_match_tau2 |
                             ~mask_match_tau1 &  mask_match_tau2]

two_matches2tau = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
two_matches1tau1e = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
two_matches1tau1mu = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
two_matches2e = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
one_match1tau = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
one_match1e = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
one_match1mu = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))
all_events = Hist(hist.axis.Regular(n_bins, lower_border, upper_border, name="", label=""))

two_matches2tau.fill(func(events_2matched_2tau.run3_dnn_moe_hh), weight =events_2matched_2tau.event_weight)
two_matches1tau1e.fill(func(events_2matched_1tau_1e.run3_dnn_moe_hh), weight =events_2matched_1tau_1e.event_weight)
two_matches1tau1mu.fill(func(events_2matched_1tau_1mu.run3_dnn_moe_hh), weight =events_2matched_1tau_1mu.event_weight)
two_matches2e.fill(func(events_2matched_2e.run3_dnn_moe_hh), weight =events_2matched_2e.event_weight)
one_match1tau.fill(func(events_1matchedtau.run3_dnn_moe_hh), weight =events_1matchedtau.event_weight)
one_match1e.fill(func(events_1match_e_fake.run3_dnn_moe_hh), weight =events_1match_e_fake.event_weight)
one_match1mu.fill(func(events_1match_mu_fake.run3_dnn_moe_hh), weight =events_1match_mu_fake.event_weight)
all_events.fill(func(events_dl.run3_dnn_moe_hh), weight =events_dl.event_weight)

# plot
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, two_matches2tau.values(), alpha=0.9, label=r"2 matched $\tau$", color='green')
ax1.step(x, two_matches1tau1e.values(), alpha=0.9, label=r"2 matches: 1 $\tau$, 1 $e$ fakes $\tau$", color='blue')
ax1.step(x, two_matches1tau1mu.values(), alpha=0.9, label=r"2 matches: 1 $\tau$, 1 $\mu$ fakes $\tau$", color='tab:pink')
ax1.step(x, two_matches2e.values(), alpha=0.9, label=r"2 matches: 2 $e$ fake $\tau$", color='tab:orange')
ax1.step(x, one_match1tau.values(), alpha=0.9, label=r"One matched $\tau$", color='tab:purple')
ax1.step(x, one_match1e.values(), alpha=0.9, label=r"One matched $e$ fakes $\tau$", color='tab:brown')
ax1.step(x, one_match1mu.values(), alpha=0.9, label=r"One matched $\mu$ fakes $\tau$", color='mediumseagreen')
ax1.step(x, all_events.values(), alpha=0.9, label=r"All events", color='red')

ax1.tick_params(axis='y', labelcolor='black')
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(fr"HH output node; tt background events, di-leptonic decay, tautau channel split in nb of $\tau$ fakes and number of reco $\tau$ matched to gen W children (matching criterion: $\Delta R < ${delr_cut_tau})", wrap=True)
plt.savefig("analysis_tautau/dnn_tau_matching_dl", dpi=300, bbox_inches='tight')
plt.show()


two_matches2tau.reset()
two_matches1tau1e.reset()
two_matches1tau1mu.reset()
two_matches2e.reset()
one_match1tau.reset()
one_match1e.reset()
one_match1mu.reset()

##########################################################################################################
##########################################################################################################
# sl decay: tau matching
had_mask = abs(events_sl.gen_top_w_children_pdgId) < 7
had_w_eta = ak.flatten(events_sl.gen_top_w_children_eta[had_mask], axis=-1)
had_w_phi = ak.flatten(events_sl.gen_top_w_children_phi[had_mask], axis=-1)

delr_qq = deltaR(
    had_w_eta[:, 0],
    had_w_phi[:, 0],
    had_w_eta[:, 1],
    had_w_phi[:, 1],
)
events_fatjet = events_sl[delr_qq < 0.6]
events_2jets  = events_sl[delr_qq >= 0.6]

W_had_eta = ak.flatten(events_2jets.gen_top_w_children_eta[abs(events_2jets.gen_top_w_children_pdgId) < 7], axis=-1)
W_had_phi = ak.flatten(events_2jets.gen_top_w_children_phi[abs(events_2jets.gen_top_w_children_pdgId) < 7], axis=-1)

# match the two single quarks to reco taus
delr1_tau = deltaR(
    events_2jets.tau_eta[:],
    events_2jets.tau_phi[:],
    events_2jets.W_had_eta[:, 0, 0],
    events_2jets.W_had_phi[:, 0, 0],
)
delr2_tau = deltaR(
    events_2jets.tau_eta[:],
    events_2jets.tau_phi[:],
    events_2jets.W_had_eta[:, 0, 1],
    events_2jets.W_had_phi[:, 0, 1],
)
delr3_tau = deltaR(
    events_2jets.tau_eta[:],
    events_2jets.tau_phi[:],
    events_2jets.W_had_eta[:, 1, 0],
    events_2jets.W_had_phi[:, 1, 0],
)
delr4_tau = deltaR(
    events_2jets.tau_eta[:],
    events_2jets.tau_phi[:],
    events_2jets.W_had_eta[:, 1, 1],
    events_2jets.W_had_phi[:, 1, 1],
)

min_delr_tau1 = np.minimum(delr1_tau, delr2_tau) # first W
min_delr_tau2 = np.minimum(delr3_tau, delr4_tau) # second W
delta_rs = np.stack([min_delr_tau1, min_delr_tau2], axis=1)
delta_rs = ak.Array(delta_rs)

# find good cut value for tau matching by looking at the delR distribution
all_delrs = ak.concatenate([ak.flatten(delr1_tau),
                            ak.flatten(delr2_tau),
                            ak.flatten(delr3_tau),
                            ak.flatten(delr4_tau)])

alldelr_max = 4.5

alldelr = Hist(hist.axis.Regular(n_bins, 0, alldelr_max, name="", label="delta R"))
alldelr.fill(all_delrs)

x = np.linspace(0, alldelr_max, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
xticks = (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
fig = plt.figure(figsize=(10, 6))
plt.bar(x, alldelr.values(), width=(alldelr_max)/n_bins, bottom=None, fill=True,  color='pink', edgecolor='black')
plt.xticks(xticks)
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title(r"Delta R of all reconstructed taus with gen W children, $\tau\tau$ channel, sl decay")

plt.savefig(f"analysis_tautau/sl_alldelrs_tau_distribution", dpi=300, bbox_inches='tight')
plt.bar(x, alldelr.values(), width=(alldelr_max)/n_bins, bottom=None, fill=True,  color='pink', edgecolor='black')
plt.xticks(xticks)
plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.show()
alldelr.reset()
from IPython import embed; embed(header="MESSAGE Line 346 | File: analysis_tautau.py")

##################################################################################################################
delta_w1_close_first_child_mask = (delr1_tau < delr2_tau) # shape [(tau1, w1child1), (tau2, w1child1)]
delta_w1_close_second_child_mask = ~delta_w1_close_first_child_mask # shape [(tau1, w1child2), (tau2, w1child2)]
delta_w2_close_first_child_mask = (delr3_tau < delr4_tau)
delta_w2_close_second_child_mask = ~delta_w2_close_first_child_mask

tau_matches_w1 = ak.concatenate((delta_w1_close_first_child_mask, delta_w1_close_second_child_mask), axis=-1)
tau_matches_w2 = ak.concatenate((delta_w2_close_first_child_mask, delta_w2_close_second_child_mask), axis=-1)
tau_matches_w1 = ak.unflatten(tau_matches_w1, 2, axis=1) # shape [[(tau1, w1child1), (tau2, w1child1)],[(tau1, w1child2), (tau2, w1child2)]]
tau_matches_w2 = ak.unflatten(tau_matches_w2, 2, axis=1) # shape [[(tau1, w2child1), (tau2, w2child1)],[(tau1, w2child2), (tau2, w2child2)]]

# re-build arrays in such a way that shape is [[w1 child1, w1 child2],[w2 child 1, w1 child2]] for each tau
tau1_w1_c1_mask = delta_w1_close_first_child_mask[:, 0]
tau1_w1_c2_mask = delta_w1_close_second_child_mask[:, 0]
tau1_w2_c1_mask = delta_w2_close_first_child_mask[:, 0]
tau1_w2_c2_mask = delta_w2_close_second_child_mask[:, 0]
tau1_w1_mask = ak.concatenate((ak.unflatten(tau1_w1_c1_mask, 1, axis=0), ak.unflatten(tau1_w1_c2_mask, 1, axis=0)), axis=-1)
tau1_w2_mask = ak.concatenate((ak.unflatten(tau1_w2_c1_mask, 1, axis=0), ak.unflatten(tau1_w2_c2_mask, 1, axis=0)), axis=-1)
tau1_mask = ak.unflatten(ak.concatenate((tau1_w1_mask, tau1_w2_mask), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]

tau2_w1_c1_mask = delta_w1_close_first_child_mask[:, 1]
tau2_w1_c2_mask = delta_w1_close_second_child_mask[:, 1]
tau2_w2_c1_mask = delta_w2_close_first_child_mask[:, 1]
tau2_w2_c2_mask = delta_w2_close_second_child_mask[:, 1]
tau2_w1_mask = ak.concatenate((ak.unflatten(tau2_w1_c1_mask, 1, axis=0), ak.unflatten(tau2_w1_c2_mask, 1, axis=0)), axis=-1)
tau2_w2_mask = ak.concatenate((ak.unflatten(tau2_w2_c1_mask, 1, axis=0), ak.unflatten(tau2_w2_c2_mask, 1, axis=0)), axis=-1)
tau2_mask = ak.unflatten(ak.concatenate((tau2_w1_mask, tau2_w2_mask), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]
# do the same for delr
delr_tau1_w1_c1 = delr1_tau[:, 0]
delr_tau1_w1_c2 = delr2_tau[:, 0]
delr_tau1_w2_c1 = delr3_tau[:, 0]
delr_tau1_w2_c2 = delr4_tau[:, 0]
delr_tau1_w1 = ak.concatenate((ak.unflatten(delr_tau1_w1_c1, 1, axis=0), ak.unflatten(delr_tau1_w1_c2, 1, axis=0)), axis=-1)
delr_tau1_w2 = ak.concatenate((ak.unflatten(delr_tau1_w2_c1, 1, axis=0), ak.unflatten(delr_tau1_w2_c2, 1, axis=0)), axis=-1)
delr_tau1 = ak.unflatten(ak.concatenate((delr_tau1_w1, delr_tau1_w2), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]

delr_tau2_w1_c1 = delr1_tau[:, 1]
delr_tau2_w1_c2 = delr2_tau[:, 1]
delr_tau2_w2_c1 = delr3_tau[:, 1]
delr_tau2_w2_c2 = delr4_tau[:, 1]
delr_tau2_w1 = ak.concatenate((ak.unflatten(delr_tau2_w1_c1, 1, axis=0), ak.unflatten(delr_tau2_w1_c2, 1, axis=0)), axis=-1)
delr_tau2_w2 = ak.concatenate((ak.unflatten(delr_tau2_w2_c1, 1, axis=0), ak.unflatten(delr_tau2_w2_c2, 1, axis=0)), axis=-1)
delr_tau2 = ak.unflatten(ak.concatenate((delr_tau2_w1, delr_tau2_w2), axis=-1), 2, axis=1) # shape [[w1 child1, w1 child2],[w2 child 1, w1 child2]]
mask_match_tau1 = ak.any(ak.any(delr_tau1 < delr_cut_tau, axis=-1), axis=-1)
mask_match_tau2 = ak.any(ak.any(delr_tau2 < delr_cut_tau, axis=-1), axis=-1)

matched_delr_tau1 = delr_tau1[mask_match_tau1]
matched_delr_tau2 = delr_tau2[mask_match_tau2]
#
mask_tau_event = ak.any(ak.any(abs(events_sl.gen_top_w_children_pdgId) == 15, axis=2), axis=1)
mask_e_event   = ak.any(ak.any(abs(events_sl.gen_top_w_children_pdgId) == 11, axis=2), axis=1)
mask_mu_event  = ak.any(ak.any(abs(events_sl.gen_top_w_children_pdgId) == 13, axis=2), axis=1)
mask_two_taus   = ak.all(ak.any(abs(events_sl.gen_top_w_children_pdgId) == 15, axis=2), axis=1)
mask_two_es   = ak.all(ak.any(abs(events_sl.gen_top_w_children_pdgId) == 11, axis=2), axis=1)


events_2matched_2tau     = events_sl[mask_match_tau1 & mask_match_tau2 & mask_two_taus]
events_2matched_1tau_1e  = events_sl[mask_match_tau1 & mask_match_tau2 & mask_e_event & mask_tau_event]
events_2matched_1tau_1mu = events_sl[mask_match_tau1 & mask_match_tau2 & mask_mu_event & mask_tau_event]
events_2matched_2e       = events_sl[mask_match_tau1 & mask_match_tau2 & mask_two_es]

events_1matchedtau = events_dl[ mask_match_tau1 & ~mask_match_tau2 & ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 15, axis=1)|# first tau candidate is match and real tau
                               ~mask_match_tau1 & mask_match_tau2   &ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 15, axis=1)]# 2nd tau candidate is match and real tau
events_1match_e_fake = events_dl[ mask_match_tau1 & ~mask_match_tau2 & ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 11, axis=1)|# first tau candidate is match and is e
                               ~mask_match_tau1 & mask_match_tau2   &ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 11, axis=1)]# 2nd tau candidate is match and is e
events_1match_mu_fake = events_dl[ mask_match_tau1 & ~mask_match_tau2 & ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 0]) == 13, axis=1)|# first tau candidate is match and is e
                               ~mask_match_tau1 & mask_match_tau2   &ak.any(abs(events_dl.gen_top_w_children_pdgId[:, 1]) == 13, axis=1)]# 2nd tau candidate is match and is e
# events_no_matches = events_dl[~mask_match_tau1 & ~mask_match_tau2] # dont exist

delr_1match      = delr_tau1[ mask_match_tau1 & ~mask_match_tau2 |
                             ~mask_match_tau1 &  mask_match_tau2]
delr2_1match      = delr_tau2[ mask_match_tau1 & ~mask_match_tau2 |
                             ~mask_match_tau1 &  mask_match_tau2]
events_1match   = events_dl[ mask_match_tau1 & ~mask_match_tau2 |
                             ~mask_match_tau1 &  mask_match_tau2]
