import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataframe_image as dfi
from pathlib import Path
import json
from IPython import embed


cache_hash = "c0f733215f"
eras = ["22pre", "22post", "23pre", "23post"]
cache_dir = Path("/data/dust/user/hergesk/HH_DNN/cache") / cache_hash

data = {}
for era in eras:
    with open(cache_dir / (era + "_pid_sizes.json"), "r") as file:
        data[era] = json.load(file)
df = pd.DataFrame(data)

df = df.rename(index={
    '1100': 'tt sl',
    '1200': 'tt dl',
    '1300': 'tt fh',
    '21101': 'hh'
})
# now merge all dy entries
mask = df.index.astype(str).str.startswith('5')
df.loc['dy'] = df.loc[mask].sum()
mask = df.index.astype(str).str.startswith('5')
df = df.loc[~mask]

# plot
dfi.export(df, "cache_statistics.png", dpi=300)
