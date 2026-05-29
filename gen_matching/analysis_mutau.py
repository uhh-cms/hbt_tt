import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script analyses the mutau channel andmatches mu, tau to gen W children.
We focus on the semi-leptonic and di-leptinic channels, as mutau is only possible there.
For the different cases, we identify if the muons and taus are fakes or not, with the aim of
analysing the different cases concerning their HH DNN output score distribution.
"""
n_bins = 100

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
from IPython import embed; embed(header="MESSAGE Line 34 | File: analysis_mutau.py")
delr_cut_tau = 0.3 # matched only if distance is smaller than delr_cut
events_tau_train = events_tt_train[events_tt_train.gen_top_w_children_pdgId == 13] # only consider muon events
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
