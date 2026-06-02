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

eps = 0#1e-6 # set eps=0 for normal scale
lower_border = 0#-14# set to 0 for lin scale
upper_border = 1#12# set to 1 for lin scale
def logit(x):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, lower_border, upper_border-eps)
def inverse_logit(y):
    x = 1 / (1 + np.exp(-y))
    return np.clip(x, eps, 1 - eps)
def identity(x):
    return x
func = identity

events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/background_characterization/20260504/tt_22pre_v14.parquet")
# events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")
events_tt = events_tt[events_tt.run3_dnn_moe_hh > 0]
events_tt = events_tt[events_tt.channel_id == 2] # mutau channel
events_tt_train = ak.concatenate([events_tt[:10000], events_tt[844445:854446]]) # first ev are dl, second sl
events_tt_train = ak.concatenate([events_tt_train, events_tt[844127:844444,]]) # also add fh events
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
# muon matching
# array with all delrs, to not loose the indices
delta_rs1 = np.column_stack([delr1_emu, delr2_emu])
delta_rs2 = np.column_stack([delr3_emu, delr4_emu])
delta_rs = np.stack([delta_rs1, delta_rs2], axis=1)
delta_rs = ak.Array(delta_rs)

# event-wise loop to find events which definitely hae one genmatched tau
# vectorised way is very difficult because indices get lost very easily
matched_mu_indices = []
w_mu_indices = []
flat_delrs = ak.flatten(delta_rs, axis=-1)
flat_pdgids = ak.flatten(events_tt_train.gen_top_w_children_pdgId, axis=-1)
for i in range(len(events_tt_train)):
        for j in range(3):
            if flat_delrs[i][j] < delr_cut_mu and abs(flat_pdgids[i][j]) == 13:
                # print(f"Event {i} has a gen-matched muon with pdgId {flat_pdgids[i][j]} and delta R {flat_delrs[i][j]}")
                matched_mu_indices.append(i)
                if j < 2:
                    w_mu_indices.append(0) # muon from first W
                else:
                    w_mu_indices.append(1) # muon from second W
                pass
unmatched_mu_indices = [i for i in range(len(events_tt_train)) if i not in matched_mu_indices]

matched_mu_events = events_tt_train[matched_mu_indices]
fake_mu_events = events_tt_train[unmatched_mu_indices]

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
all_events.fill(func(events_tt.run3_dnn_moe_hh), weight =events_tt.event_weight)

# plot histograms
x = np.linspace(lower_border, upper_border, n_bins + 1)  # bin edges
x = (x[:-1] + x[1:]) / 2  # bin centers

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(right=0.85)
ax1.set_xlabel('HH output node')
ax1.set_ylabel('Number of events', color="black")
ax1.step(x, list(mu_events_dl.values()), alpha=0.9, label=r'matched dl mu events', color='green')
ax1.step(x, list(mu_events_sl.values()), alpha=0.9, label=r'matched sl mu events', color='mediumseagreen')
ax1.step(x, list(mu_fakes_dl.values()), alpha=0.9, label=r'fake dl mu events', color='red')
ax1.step(x, list(mu_fakes_sl.values()), alpha=0.9, label=r'fake sl mu events', color='slateblue')
ax1.step(x, list(mu_fakes_fh.values()), alpha=0.9, label=r'fake fh mu events', color='purple')

ax1.step(x, list(all_events.values()), label='all events in mutau channel', color='blue')
# ax1.fill_between(x, list(all_events.values()), color='red', alpha=0.1)

ax1.tick_params(axis='y', labelcolor="black")
ax1.get_legend_handles_labels()
plt.legend()
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylim(bottom=1e-1)
fig.tight_layout()
plt.title(f"HH output node; mutau channel tt bg split in correctly matched and fake muon events (matching criterion: $\Delta R < {delr_cut_mu}$)")
plt.savefig("analysis_mutau/dnn_mu_matching", dpi=300, bbox_inches='tight')
plt.show()

####################################################################################################################################
# semi-leptonic case: tau matching
# for the semi-leptonic case, if the mu is correct, the tau HAS TO BE FAKE bc the other W needs to decay hadronically,
# with the jets emerging from the hadronic decaying W being misidentified as tau_had

# find the indices of the hadronic decaying W, which is the one leading to the fake tau
invert_indices = lambda x: 1 if x == 0 else 0
had_w_indices = []
had_w_indices = [invert_indices(i) for i in w_mu_indices]
from IPython import embed; embed(header="MESSAGE Line 204 | File: analysis_mutau.py")
# useful columns
# matched_mu_sl.tau_eta
# matched_mu_sl.tau_phi
# gen_top_w_children_eta
# gen_top_w_children_phi
# gen_top_w_children_pdgId
# gen_top_w_eta
# gen_top_w_phi
