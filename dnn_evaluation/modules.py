import numpy as np
import torch

def logit(x, eps=1e-6, lower_border=-14, upper_border=12):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, lower_border, upper_border-eps)
def inverse_logit(y):
    x = 1 / (1 + np.exp(-y))
    return np.clip(x, eps, 1 - eps)
def identity(x):
    return x

def asimov_significance(s, *b):
    """
    Asimov Significance.
    Approximation coming from asimov for no background uncertainty: https://arxiv.org/abs/1806.00322 eq. 3.2
    This approximation is unstable for two cases, and thus certain epsilons are introduced to stabilize  it.:
    It is unstable for no-background (b=0) regions, which is why *eps_b* is chosen as default value.
    But also when s > (s+b)(ln(1+s/b)), for which *eps_s* increase stability and lower bound the significance.

    Args:
        s (Hist): Histogram representing signal in bin.
        b (Hist): Histogram representing background in bin.
        eps_b (int, optional):background uncertainty. Defaults to 1 to prevent very high sig values.
        eps_s (_type_, optional): signal uncertainty. Defaults to 1e-9.

    Returns:
        numpy.ndarray: Asimov Significance with background uncertainty
    """
    eps_s = 1e-9
    eps_b = 1e-9
    # from IPython import embed; embed(header="MESSAGE Line 33 | File: modules.py")
    s_count = s.values()
    s_error = np.sqrt(s_count)
    # for background, negative weights can exist, which is why they are set to 0 for the significance calculation
    b_count = []
    bs_error = 0
    for b_hist in b:
        _b = b_hist.values()
        neg_mask = _b < 0
        _b = np.where(neg_mask, 0, _b)
        b_count.append(_b)
        bs_error += np.sqrt(_b)
    b_count = np.sum(b_count, axis=0)
    b_error = np.sum(bs_error)
    # sig² for simple sig function:
    # sig_per_bin = s_count**2 / (b_count + eps)
    # asimov sig²:
    s_count = s_count + eps_s
    b_count = b_count + eps_b
    sigsquared_per_bin = 2 * ((s_count + b_count) * np.log(1 + s_count / (b_count )) - s_count) # asimov sig fct
    sig_per_bin = np.sqrt(np.abs(sigsquared_per_bin))
    error_per_bin = np.sqrt((np.log(s_count/b_count + 1)*s_error/sig_per_bin)**2 + (((np.log(s_count/b_count+1)*b_count - s_count)/b_count)*b_error/sig_per_bin)**2)
    return sig_per_bin, error_per_bin

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
