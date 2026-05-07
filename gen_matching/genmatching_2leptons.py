import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the leptons emerging from the W decay.
"""
n_bins = 50

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
events_tt_train = events_tt[events_tt.run3_dnn_moe_hh > 0]#[:100000]

# important columns
# events_tt_train.gen_top_w_children_eta
# events_tt_train.gen_top_w_children_phi
# events_tt_train.emu_eta
# events_tt_train.emu_phi
# events_tt_train.tau_eta
# events_tt_train.tau_phi

def deltaR(eta1, phi1, eta2, phi2):
    delta_eta = eta2 - eta1
    delta_phi = phi2 - phi1
    # Ensure delta_phi is in the range [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

delr_cut = 0.05 # matched only if distance is smaller than delr = 0.05
# emu gen matching
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

# # some of the tau events have two entries
# indices = ak.where(ak.count(events_tt_train.tau_eta, axis=1)==2)
# sliced_tau_eta = events_tt_train.tau_eta[indices]
# sliced_tau_phi = events_tt_train.tau_phi[indices]
# sliced_W_eta = events_tt_train.gen_top_w_children_eta[indices]
# sliced_W_phi = events_tt_train.gen_top_w_children_phi[indices]
# delr5_tau = deltaR(
#     sliced_tau_eta[:, 1],
#     sliced_tau_phi[:, 1],
#     sliced_W_eta[:, 0, 0],
#     sliced_W_phi[:, 0, 0],
# )
# delr6_tau = deltaR(
#     sliced_tau_eta[:, 1],
#     sliced_tau_phi[:, 1],
#     sliced_W_eta[:, 0, 1],
#     sliced_W_phi[:, 0, 1],
# )
# delr7_tau = deltaR(
#     sliced_tau_eta[:, 1],
#     sliced_tau_phi[:, 1],
#     sliced_W_eta[:, 1, 0],
#     sliced_W_phi[:, 1, 0],
# )
# delr8_tau = deltaR(
#     sliced_tau_eta[:, 1],
#     sliced_tau_phi[:, 1],
#     sliced_W_eta[:, 1, 1],
#     sliced_W_phi[:, 1, 1],
# )

# match emu to smallest distance gen top W children
min_delr_emu1 = np.minimum(delr1_emu, delr2_emu)
min_delr_emu2 = np.minimum(delr3_emu, delr4_emu)
min_delr_emu = np.minimum(min_delr_emu1, min_delr_emu2)
delta_rs_emu = ak.Array(min_delr_emu)

# match tau to smallest distance gen top W children
min_delr_tau1 = np.minimum(delr1_tau, delr2_tau)
min_delr_tau2 = np.minimum(delr3_tau, delr4_tau)
min_delr_tau = np.minimum(min_delr_tau1, min_delr_tau2) # matched tau events with only one entry
delta_rs_tau = ak.Array(min_delr_tau)
# min_delr_tau3 = np.minimum(delr5_tau, delr6_tau)
# min_delr_tau4 = np.minimum(delr7_tau, delr8_tau)
# min_delr_tau_2 = np.minimum(min_delr_tau3, min_delr_tau4) # matched tau events, second entry if it exists

# merge the two arrays for tau events with one or two entries
# min_delr_tau = ak.full_like(events_tt_train.tau_eta, [10, 10])
# min_delr_tau[:,0] = min_delr_tau_1
# min_delr_tau[indices][:,1] = min_delr_tau_2

# matched only if delta r smaller than delr_cut
delr_matched_emu = delta_rs_emu[delta_rs_emu < delr_cut]
delr_matched_tau = delta_rs_tau[delta_rs_tau < delr_cut]
print("delta r matched emu:", delr_matched_emu)
print("delta r matched tau:", delr_matched_tau)
