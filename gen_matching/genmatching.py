import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

"""This script performs a Gen Matching for the b quarks emerging from the tt background.
"""

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260429/tt_22pre_v14.parquet")
events_tt_train = events_tt[:100]

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

delta_rs = []
closest_b = []
for i in range(len(events_tt_train)):
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


from IPython import embed; embed(header="MESSAGE Line 44 | File: genmatching.py")
# TODO find an alternative to that mask that works
# mask = ak.any(delta_rs < 0.1)
# delta_rs = delta_rs[mask]
# closest_b = closest_b[mask]
print(delta_rs)
