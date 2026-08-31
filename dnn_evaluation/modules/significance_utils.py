import numpy as np
import torch

from hist import Hist
from dataclasses import dataclass
from pathlib import Path
import awkward as ak
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

import sys
sys.path.append("/afs/desy.de/user/h/hergesk/repos/hbt_tt/dnn_evaluation/modules")
from modules import logit
from IPython import embed

"""
This script features all important classes and functions for the different significance plots.
"""

class SigLoader:
    """
    calculates and stores the 3x3 tt background significance values for 
    - 3 categories: etau, mutau, tautau
    - times 3 W decay modes: dl, sl, fh
    
    input: significances of baseline DNN (dict)
    """
    def __init__(self, baseline: pd.DataFrame):
        self.baseline = baseline
            
    def make_matrix(self, sigs: pd.DataFrame, save_path: str):
        """calculates a sig matrix from the input dict.
        calculates the difference to the baseline values
        and plots it as a colormap."""
        fig, ax = plt.subplots(figsize=(5, 4))
        cell_color = "#B382C2"
        # Cells
        cmap = LinearSegmentedColormap.from_list(
        "blue_rose",
        ["#4A90E2", "#FFFFFF", "#D98BA3"]
)
        for row in range(3):
            for col in range(3):
                # compute diff between sigs and self.baseline
                # higher sig is red, lower blue
                diff = sigs.iloc[row, col] - self.baseline.iloc[row, col]
            
                # normalize diff so that 0 is middle -> white
                norm = Normalize(vmin=-0.12, vmax=0.12)
                cell_color = cmap(norm(diff))

                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5), 1, 1, facecolor=cell_color, alpha=0.25, edgecolor="0.65", linewidth=0.8)
                )
                ax.text(
                    col, row, 
                    f"{sigs.iloc[row, col]:.4f}", # take corresponding sig from dataframe and round to exactly 4 decimal nb 
                    ha="center", va="center", fontsize=12
                )

        for j, col in zip([0,1,2], [r"$e\tau$", r"$\mu\tau$", r"$\tau\tau$"]):
            ax.text(
                (j + 0.5) / 3, 1.04, col, transform=ax.transAxes, ha="center", va="bottom", fontsize=11
            )

        ax.set_yticks(range(3))
        ax.set_yticklabels(["tt dl", "tt sl", "tt fh"], fontsize=11)
        ax.set_xticks([]) # no x ticks as i did this manually in the loop above
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label=r"$\Delta\text{ } Z_A$ to baseline")

        ax.tick_params(which="both", length=0)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
    