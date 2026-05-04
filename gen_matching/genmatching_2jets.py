import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the b quarks emerging from the tt background.
"""
n_bins = 20

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
events_tt_train = events_tt[:1000]

# important columns
# events_tt_train.bjet_eta
# events_tt_train.bjet_phi
# events_tt_train.bjet_btag
# events_tt.gen_top_b_eta
# events_tt.gen_top_b_phi

def deltaR(eta1, phi1, eta2, phi2):
    delta_eta = eta2 - eta1
    delta_phi = phi2 - phi1
    # Ensure delta_phi is in the range [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

delr_cut = 0.05 # matched only if distance is smaller than delr = 0.05

delta_rs = []

print("Number of events: ", len(events_tt_train))
for i in range(len(events_tt_train)):
    # instead of looping, next time use events_tt_train.bjet_eta[:,:2] etc and apply function to whole array
    if i % (len(events_tt_train) // 5) == 0 and i != 0:
        print("Processing event", i)
    # match bjet 1 to gen top bs
    delta_r1 = deltaR(events_tt_train.bjet_eta[i][0], events_tt_train.bjet_phi[i][0],
                      events_tt_train.gen_top_b_eta[i][0], events_tt_train.gen_top_b_phi[i][0])
    delta_r2 = deltaR(events_tt_train.bjet_eta[i][1], events_tt_train.bjet_phi[i][1],
                      events_tt_train.gen_top_b_eta[i][0], events_tt_train.gen_top_b_phi[i][0])
    # match bjet2 to gen top bs
    delta_r3 = deltaR(events_tt_train.bjet_eta[i][0], events_tt_train.bjet_phi[i][0],
                      events_tt_train.gen_top_b_eta[i][1], events_tt_train.gen_top_b_phi[i][1])
    delta_r4 = deltaR(events_tt_train.bjet_eta[i][1], events_tt_train.bjet_phi[i][1],
                      events_tt_train.gen_top_b_eta[i][1], events_tt_train.gen_top_b_phi[i][1])
    # append smallest deltar value
    delta_rs.append([np.minimum(delta_r1, delta_r2), np.minimum(delta_r3, delta_r4)])

delta_rs = ak.Array(delta_rs)

mask = delta_rs < delr_cut

delta_rs = delta_rs[mask]
btags_of_matched_events = events_tt_train.bjet_btag[mask]
# ev_tt_obj_indices = ak.local_index(events_tt_train.bjet_eta)[mask]

print("Delta rs:", delta_rs)

delr_hist = ak.flatten(delta_rs, axis=None)
delr = Hist(hist.axis.Regular(n_bins, 0, delr_cut, name="", label="delta_r"))
delr.fill(delr_hist)

x = np.linspace(0, delr_cut, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig = plt.figure(figsize=(10, 6))
plt.bar(x, delr.values(), width=(delr_cut)/n_bins, bottom=None, fill=True,  color='pink', edgecolor='black')#, label=f'hh x ({scaling_factor:.2f})')

plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of both gen b jets with matched gen top b quark")

plt.savefig(f"images/delr_2jets_hist", dpi=300, bbox_inches='tight')
plt.show()
delr.reset()

# define three event classes
# 1: both b jets matched to gen top b quarks
two_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(btags_of_matched_events, axis = 1) == 2)]
# 2: only one b jet matched to gen top b quark
one_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(btags_of_matched_events, axis = 1) == 1)]
# 3: no b jet matched to gen top b quark
no_matched = events_tt_train.run3_dnn_moe_hh[ak.where(ak.count(btags_of_matched_events, axis = 1) == 0)]

# plot the btag output score hists
eps = 0#1e-6 # set eps=0 for normal scale
endpoint = 1#5 # set to 1 for normal scale
def logit(x):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, -14, 5-eps)
def identity(x):
    return x
func = identity


two_matched_hist = Hist(hist.axis.Regular(n_bins, func(eps), endpoint, name="2_matched", label="2_matched"))
one_matched_hist = Hist(hist.axis.Regular(n_bins, func(eps), endpoint, name="1_matched", label="1_matched"))
no_matched_hist = Hist(hist.axis.Regular(n_bins, func(eps), endpoint, name="no_matched", label="no_matched"))

two_matched_hist.fill(func(two_matched), weight =events_tt_train.event_weight[ak.where(ak.count(btags_of_matched_events, axis = 1) == 2)])
one_matched_hist.fill(func(one_matched), weight =events_tt_train.event_weight[ak.where(ak.count(btags_of_matched_events, axis = 1) == 1)])
no_matched_hist.fill(func(no_matched), weight =events_tt_train.event_weight[ak.where(ak.count(btags_of_matched_events, axis = 1) == 0)])

# plot histograms
x = np.linspace(0, 1, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig = plt.figure(figsize=(10, 6))

for hist, label, color in zip([two_matched_hist, one_matched_hist, no_matched_hist],
                             ['Two matched', 'One matched', 'No matched'],
                             ['green', 'orange', 'red']):
    plt.bar(
        x,
        hist.values(),
        width=(func(eps)-func(endpoint))/n_bins,
        bottom=None,
        fill=False,
        label=label,
        color=color,
        edgecolor='black'
    )

    plt.xlabel("logit of HH output node")
    plt.ylabel("Number of events")

    plt.title("HH output node for two gen matched b jets")
    plt.legend(f"{label} b jets")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    plt.savefig(f"images/hh_output_{label.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')

    hist.reset()
