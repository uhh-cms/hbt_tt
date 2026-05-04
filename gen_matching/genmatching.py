import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the b quarks emerging from the tt background.
"""
n_bins = 20

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260429/tt_22pre_v14.parquet")
events_tt_train = events_tt[:100000]

# important columns
# events_tt.gen_top_b_eta
# events_tt.gen_top_b_phi
# events_tt.jet1_eta
# events_tt.jet1_phi
# events_tt.n_jet

def deltaR(eta1, phi1, eta2, phi2):
    delta_eta = eta2 - eta1
    delta_phi = phi2 - phi1
    # Ensure delta_phi is in the range [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)
delr_cut = 0.05 # matched only if distance is smaller than delr = 0.05

delta_rs = []
closest_b = []
print("Number of events: ", len(events_tt_train))
for i in range(len(events_tt_train)):
    if i % (len(events_tt_train) // 5) == 0 and i != 0:
        print("Processing event", i)
    delta_r1 = deltaR(events_tt_train.gen_top_b_eta[i][0], events_tt_train.gen_top_b_phi[i][0],
                      events_tt_train.jet1_eta[i], events_tt_train.jet1_phi[i])
    delta_r2 = deltaR(events_tt_train.gen_top_b_eta[i][1], events_tt_train.gen_top_b_phi[i][1],
                      events_tt_train.jet1_eta[i], events_tt_train.jet1_phi[i])
    if delta_r1 < delta_r2:
        delta_rs.append(delta_r1)
        closest_b.append(0)
    elif delta_r1 > delta_r2:
        delta_rs.append(delta_r2)
        closest_b.append(1)
    elif delta_r1 == delta_r2:
        delta_rs.append(delta_r1)
        closest_b.append(3)

delta_rs = ak.Array(delta_rs)
closest_b = ak.Array(closest_b)

mask = delta_rs < delr_cut

delta_rs = delta_rs[mask]
closest_b = closest_b[mask]
print(delta_rs)

delr = Hist(hist.axis.Regular(n_bins, 0, delr_cut, name="", label="delta_r"))
delr.fill(delta_rs)

x = np.linspace(0, delr_cut, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers
fig = plt.figure(figsize=(10, 6))
plt.bar(x, delr.values(), width=(delr_cut)/n_bins, bottom=None, fill=True,  color='pink', edgecolor='black')#, label=f'hh x ({scaling_factor:.2f})')

plt.xlabel("delta R = $\sqrt{\Delta \eta² + \Delta \phi²}$")
plt.ylabel("Number of events")
plt.title("Delta R of gen b quark and matched jet")

plt.savefig(f"images/delr_hist", dpi=300, bbox_inches='tight')
plt.show()
delr.reset()
