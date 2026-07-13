import numpy as np
import torch


def def_equbin(
    in_distr: torch.tensor,
    binsize=None,
    bin_num: int=100,
    hist_edge_l: int=-14) -> tuple:

    """
    Flat-s binning with overflow bin on the left side. Only the number of events counts, weights do not contribute.

    Args:
        in_distr (torch.tensor): Tensor representing all signal events.
        hist_edge_l (int): Integer representing the left most edge of the binned histogram.

    Returns:
        tuple: (
        hist: 2D NumPy array of dimension (bin_num, bin_size) containing the sorted values for each regular bin,
        odd_bin: 1D array of the remaining values on the left side,
        bins_limits: 1D array containing the exact bin edges from hist_edge_l to the maximum value
        )
    """
    in_distr_filtered = in_distr[in_distr > hist_edge_l+1]
    distr_size = len(in_distr_filtered)

    bin_size = distr_size // bin_num
    odd_bin_size = distr_size % bin_num

    args = in_distr_filtered.argsort()

    hist = np.zeros((bin_num, bin_size))

    for i in range(bin_num):
        start_idx = odd_bin_size + i * bin_size
        end_idx = odd_bin_size + (i + 1) * bin_size
        hist[i, :] = in_distr_filtered[args[start_idx:end_idx]]

    if odd_bin_size == 0:
        odd_bin = None
        bins_limits = np.arange(bin_num) * bin_size
        bins_limits = args[bins_limits]
        bins_limits = np.concatenate(([hist_edge_l],in_distr_filtered[bins_limits],
                                    [in_distr_filtered[args[-1]]]))

    else:
        # odd_bin nimmt nun die ersten Elemente auf der linken Seite
        odd_bin = in_distr_filtered[args[:odd_bin_size]]

        # Die Limits starten bei 0 (für den odd_bin) und dann jeweils am Anfang der regulären Bins
        limit_indices = np.arange(bin_num) * bin_size + odd_bin_size
        limit_indices = np.concatenate(([0], limit_indices))

        bins_limits = in_distr_filtered[args[limit_indices]]
        # Letztes Limit (das Maximum) wird wie gewohnt angehängt
        bins_limits = np.concatenate((bins_limits, [in_distr_filtered[args[-1]]]))
        bins_limits[0]=hist_edge_l

    return (hist, odd_bin, bins_limits)
