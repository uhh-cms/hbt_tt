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
events_tt = events_tt[events_tt.channel_id == 2] # mutau channel
weights_2d = ak.zeros_like(events_tt.tau_genPartFlav)+events_tt.event_weight
binned_matched = np.bincount(ak.to_numpy(ak.flatten(events_tt.tau_genPartFlav)), weights=ak.flatten(weights_2d))

# # plot
# x = np.arange(6)
# labels = ["Unknown",
#         "Prompt e",
#         r"Prompt $\mu$",
#         r"$\tau_e$",
#         r"$\tau_\mu$",
#         r"$\tau_h$"]
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(x, binned_matched, color='blueviolet', alpha=0.5, edgecolor='black')
# plt.xlabel("origin")
# plt.ylabel("Number of events")
# plt.title(fr"Origin of gen matched taus in $\mu\tau$ channel (from tau_genPartFlav column)")
# # plt.xticks(x, labels, rotation=45)
# plt.yscale('linear')
# plt.savefig(fr"analysis_mutau/origin_of_genmatched_tau_taugenpartflav_weighted", dpi=300, bbox_inches='tight')
# plt.show()
from IPython import embed; embed(header="MESSAGE Line 54 | File: origin_genmatched_tau.py")
unknown_counts = ak.sum(events_tt.event_weight[ak.flatten(events_tt.tau_genPartFlav == 0)])
e_counts = ak.sum(events_tt.event_weight[ak.flatten(events_tt.tau_genPartFlav == 1)])
mu_counds = ak.sum(events_tt.event_weight[ak.flatten(events_tt.tau_genPartFlav == 2)])
tau_counts = ak.sum(events_tt.event_weight[ak.flatten(events_tt.tau_genPartFlav == 3)]) + ak.sum(events_tt.event_weight[ak.flatten(events_tt.tau_genPartFlav == 4)]) + ak.sum(events_tt.event_weight[ak.flatten(events_tt.tau_genPartFlav == 5)])

origins = ["Unknown", r"e", r"$\mu$", r"$\tau$"]
counts = [unknown_counts, e_counts, mu_counds, tau_counts]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(origins, counts, color='blueviolet', alpha=0.5, edgecolor='black')
plt.xlabel("origin")
plt.ylabel("Number of events")
plt.title(r"Origin of gen matched taus in $\mu\tau$ channel (from tau_genPartFlav column)")
# plt.xticks(x, labels, rotation=45)
plt.yscale('linear')
plt.savefig("analysis_mutau/origin_of_genmatched_tau_taugenpartflav_weighted", dpi=300, bbox_inches='tight')
plt.show()
