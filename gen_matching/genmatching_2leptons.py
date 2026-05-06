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
    events_tt_train.emu_eta[:, 0],
    events_tt_train.emu_phi[:, 0],
    events_tt_train.gen_top_w_children_eta[:, 0],
    events_tt_train.gen_top_w_children_phi[:, 0],
)

delr2_emu = deltaR(
    events_tt_train.emu_eta[:, 0],
    events_tt_train.emu_phi[:, 0],
    events_tt_train.gen_top_w_children_eta[:, 1],
    events_tt_train.gen_top_w_children_phi[:, 1],
)

delr3_emu = deltaR(
    events_tt_train.emu_eta[:, 1],
    events_tt_train.emu_phi[:, 1],
    events_tt_train.gen_top_w_children_eta[:, 0],
    events_tt_train.gen_top_w_children_phi[:, 0],
)

delr4_emu = deltaR(
    events_tt_train.emu_eta[:, 1],
    events_tt_train.emu_phi[:, 1],
    events_tt_train.gen_top_w_children_eta[:, 1],
    events_tt_train.gen_top_w_children_phi[:, 1],
)
# tau gen matching
delr1_tau = deltaR(
    events_tt_train.tau_eta[:, 0],
    events_tt_train.tau_phi[:, 0],
    events_tt_train.gen_top_w_children_eta[:, 0],
    events_tt_train.gen_top_w_children_phi[:, 0],
)

delr2_tau = deltaR(
    events_tt_train.tau_eta[:, 0],
    events_tt_train.tau_phi[:, 0],
    events_tt_train.gen_top_w_children_eta[:, 1],
    events_tt_train.gen_top_w_children_phi[:, 1],
)

delr3_tau = deltaR(
    events_tt_train.tau_eta[:, 1],
    events_tt_train.tau_phi[:, 1],
    events_tt_train.gen_top_w_children_eta[:, 0],
    events_tt_train.gen_top_w_children_phi[:, 0],
)

delr4_tau = deltaR(
    events_tt_train.tau_eta[:, 1],
    events_tt_train.tau_phi[:, 1],
    events_tt_train.gen_top_w_children_eta[:, 1],
    events_tt_train.gen_top_w_children_phi[:, 1],
)
