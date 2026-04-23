import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

"""This script plots the tt background data
with further splitting the tt background into its six different category ids.
"""
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data

etau_mask_res1b_sl = ak.any(events_tt.category_ids == 147, axis = 1) & (events_tt.process_id == 1100)
etau_mask_res1b_dl = ak.any(events_tt.category_ids == 147, axis = 1) & (events_tt.process_id == 1200)
etau_mask_res1b_fh = ak.any(events_tt.category_ids == 147, axis = 1) & (events_tt.process_id == 1300)

etau_mask_res2b_sl = ak.any(events_tt.category_ids == 151, axis = 1) & (events_tt.process_id == 1100)
etau_mask_res2b_dl = ak.any(events_tt.category_ids == 151, axis = 1) & (events_tt.process_id == 1200)
etau_mask_res2b_fh = ak.any(events_tt.category_ids == 151, axis = 1) & (events_tt.process_id == 1300)

mutau_mask_res1b_sl = ak.any(events_tt.category_ids == 175, axis = 1) & (events_tt.process_id == 1100)
mutau_mask_res1b_dl = ak.any(events_tt.category_ids == 175, axis = 1) & (events_tt.process_id == 1200)
mutau_mask_res1b_fh = ak.any(events_tt.category_ids == 175, axis = 1) & (events_tt.process_id == 1300)

mutau_mask_res2b_sl = ak.any(events_tt.category_ids == 179, axis = 1) & (events_tt.process_id == 1100)
mutau_mask_res2b_dl = ak.any(events_tt.category_ids == 179, axis = 1) & (events_tt.process_id == 1200)
mutau_mask_res2b_fh = ak.any(events_tt.category_ids == 179, axis = 1) & (events_tt.process_id == 1300)

tautau_mask_res1b_sl = ak.any(events_tt.category_ids == 203, axis = 1) & (events_tt.process_id == 1100)
tautau_mask_res1b_dl = ak.any(events_tt.category_ids == 203, axis = 1) & (events_tt.process_id == 1200)
tautau_mask_res1b_fh = ak.any(events_tt.category_ids == 203, axis = 1) & (events_tt.process_id == 1300)

tautau_mask_res2b_sl = ak.any(events_tt.category_ids == 207, axis = 1) & (events_tt.process_id == 1100)
tautau_mask_res2b_dl = ak.any(events_tt.category_ids == 207, axis = 1) & (events_tt.process_id == 1200)
tautau_mask_res2b_fh = ak.any(events_tt.category_ids == 207, axis = 1) & (events_tt.process_id == 1300)

masks = [[etau_mask_res1b_sl, etau_mask_res1b_dl, etau_mask_res1b_fh],
         [etau_mask_res2b_sl, etau_mask_res2b_dl, etau_mask_res2b_fh],
         [mutau_mask_res1b_sl, mutau_mask_res1b_dl, mutau_mask_res1b_fh],
         [mutau_mask_res2b_sl, mutau_mask_res2b_dl, mutau_mask_res2b_fh],
         [tautau_mask_res1b_sl, tautau_mask_res1b_dl, tautau_mask_res1b_fh],
         [tautau_mask_res2b_sl, tautau_mask_res2b_dl, tautau_mask_res2b_fh]]

categories = ["etau, res 1b","etau, res 2b", "mutau, res 1b", "mutau, res 2b", "tautau, res 1b", "tautau, res 2b"]

# plot grouped bar chart
x = np.arange(6)
tt_sl = []
tt_dl = []
tt_fh = []
for mask in masks:
    tt_sl.append(ak.sum(events_tt.event_weight[mask[0]]))
    tt_dl.append(ak.sum(events_tt.event_weight[mask[1]]))
    tt_fh.append(ak.sum(events_tt.event_weight[mask[2]]))

width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, tt_sl, width, label='semileptonic')
ax.bar(x, tt_dl, width, label='dileptonic')
ax.bar(x + width, tt_fh, width, label='fully hadronic')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_xlabel('category')
ax.set_ylabel('Events')
ax.set_title('subprocesses of the tt background')
ax.legend()
ax.set_yscale('log')

plt.savefig(f"plots/tt_category_barplot2.png", dpi=300, bbox_inches='tight')
plt.show()
