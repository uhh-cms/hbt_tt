import numpy as np
import torch
import functools
import operator
from termcolor import colored

def logit(x, eps=1e-6, lower_border=-14, upper_border=12):
    # set this fct to return x for normal scale
    y = np.log((x + eps) / (1 - x + eps))
    return np.clip(y, lower_border, upper_border-eps)
def inverse_logit(y):
    x = 1 / (1 + np.exp(-y))
    return np.clip(x, eps, 1 - eps)
def identity(x):
    return x

def get_error(h, error_type: str, density: bool = False):
    """
    Calculate the error to be plotted for the given histogram *h*.
    Supported error types are:

        - "variance": the plotted error is the square root of the variance for each bin
        - "poisson_unweighted": the plotted error is the poisson error for each bin
        - "poisson_weighted": the plotted error is the poisson error for each bin, weighted by the variance
    If the histogram is a density histogram, the error is scaled by the area of the histogram.
    """
    if density:
        area = functools.reduce(operator.mul, h.axes.widths)
        h = h * area

    # determine the error type
    if error_type == "variance":
        yerr = h.view().variance ** 0.5

    elif error_type in {"poisson_unweighted", "poisson_weighted"}:
        # compute asymmetric poisson confidence interval
        from hist.intervals import poisson_interval

        variances = h.view().variance if error_type == "poisson_weighted" else None
        values = h.view().value
        confidence_interval = poisson_interval(values, variances)

        # negative values are considerd as blinded bins -> set confidence interval to 0
        confidence_interval[:, values < 0] = 0

        if error_type == "poisson_weighted":
            # might happen if some bins are empty, see https://github.com/scikit-hep/hist/blob/5edbc25503f2cb8193cc5ff1eb71e1d8fa877e3e/src/hist/intervals.py#L74  # noqa: E501
            confidence_interval[np.isnan(confidence_interval)] = 0
        elif np.any(np.isnan(confidence_interval)):
            raise ValueError("Unweighted Poisson interval calculation returned NaN values, check Hist package")

        # calculate the error
        yerr_lower = values - confidence_interval[0]
        yerr_upper = confidence_interval[1] - values
        yerr = np.array([yerr_lower, yerr_upper])
        # hist name for debugging purposes:
        ax = h.axes[0]
        if np.any(yerr < 0):
            print(colored(f"found yerr < 0, forcing to 0; this should not happen, please check your histogram: {ax.name}", "red"))
            yerr[yerr < 0] = 0

    else:
        raise ValueError(f"unknown error type '{error_type}'")

    # re-apply density if needed
    if density:
        area = functools.reduce(operator.mul, h.axes.widths)
        h = h / area
        yerr = yerr / area
    return yerr


def asimov_significance(s, *b, error_type="poisson_weighted", eps_s=1e-9, eps_b=1e-9):
    """
    Asimov Significance.
    Approximation coming from asimov for no background uncertainty: https://arxiv.org/abs/1806.00322 eq. 3.2
    This approximation is unstable for two cases, and thus certain epsilons are introduced to stabilize  it.:
    It is unstable for no-background (b=0) regions, which is why *eps_b* is chosen as default value.
    But also when s > (s+b)(ln(1+s/b)), for which *eps_s* increase stability and lower bound the significance.

    Args:
        s (Hist): Histogram representing signal in bin.
        b (Hist): Histogram representing background in bin.
        error_type (str, optional): Type of error to use. Defaults to "poisson_weighted".
        eps_b (int, optional):background uncertainty. Defaults to 1 to prevent very high sig values.
        eps_s (_type_, optional): signal uncertainty. Defaults to 1e-9.

    Returns:
        numpy.ndarray: Asimov Significance with background uncertainty
    """
    eps_s = 1e-9
    eps_b = 1e-9
    eps_sig = 1e-9
    s_count = s.values()
    if np.any(s_count < 0):
        print(colored("Warning: Negative signal counts encountered. Setting them to 0.", "red"))
        s_count = np.where(s_count < 0, 0, s_count)
    s_error = np.sqrt(s_count+eps_s)
    # for background, negative weights can exist, which is why they are set to 0 for the significance calculation
    b_count = []
    for b_hist in b:
        _b = b_hist.values()
        neg_mask = _b < 0
        _b = np.where(neg_mask, 0, _b)
        b_count.append(_b)
    b_count = np.sum(b_count, axis=0)
    # implement a dynamic epsilon for background, so that sig for bg = 0 is not too high
    mask_bcount_zero = np.where(b_count == 0)
    b_count[mask_bcount_zero] = None
    epsilon_b = min(b_count)
    b_count[np.isnan(b_count)] = epsilon_b

    s_error = get_error(s, error_type=error_type)
    b_error = [get_error(_b , error_type=error_type) for _b in b]
    b_error = np.sqrt(np.sum(np.array(b_error) ** 2, axis=0))
    # to not get error bars of zero length for non-bg bins, add an error:
    b_error[b_error==0] = 1
    s_count = s_count + eps_s
    b_count = b_count + eps_b

    sigsquared_per_bin = 2 * ((s_count + b_count) * np.log(1 + s_count / (b_count )) - s_count) # asimov sig fct
    if np.any(sigsquared_per_bin < 0):
        approx_sigsquared_per_bin = (s_count** 2 / (b_count + eps_b))  # approximate sig fct for s << b
        sigsquared_per_bin = np.where(sigsquared_per_bin < 0, approx_sigsquared_per_bin, sigsquared_per_bin)
    sig_per_bin = np.sqrt(sigsquared_per_bin)+eps_sig
    error_per_bin = np.sqrt((np.log(s_count/b_count + 1)*s_error/sig_per_bin)**2 + (((np.log(s_count/b_count+1)*b_count - s_count)/b_count)*b_error/sig_per_bin)**2)
    return sig_per_bin, error_per_bin

def add_flow_bin(h, underflow: bool = True, overflow: bool = True):
    """Add under- and/or overflow bin to the histogram *h*.
    """
    h = h.values(flow=True)
    if underflow:
        h[1] += h[0]
    if overflow:
        h[-2] += h[-1]
    h_with_flow = h[1:-1]
    return h_with_flow

def flats_binning(
    sig_distr: torch.tensor,
    # binsize: int,
    bin_num: int,
    hist_edge_l: int=-1e2) -> tuple:

    """
    Flat-s binning. Only the number of events counts, weights do not contribute.
    This function bins signal events into a specified number of bins, ensuring that each bin contains an approximately equal number of events.
    The leftmost bin also includes the remainding events which were too few to distribute between all bins.

    Args:
        sig_distr (torch.tensor): Tensor representing all signal events.
        hist_edge_l (int): Integer representing the left most edge of the binned histogram.

    Returns:
        tuple: (
        hist: 2D NumPy array of dimension (bin_num, bin_size) containing the sorted values for each regular bin,
        odd_bin: 1D array of the remaining values on the left side,
        bins_limits: 1D array containing the exact bin edges from hist_edge_l to the maximum value
        )
    """
    sig_distr_filtered = sig_distr[sig_distr > hist_edge_l+1]
    nb_events = len(sig_distr_filtered)

    nb_ev_per_bin = nb_events // bin_num
    remainder_ev_per_bin = nb_events % bin_num

    args = sig_distr_filtered.argsort()

    # hist = np.zeros((bin_num, nb_ev_per_bin))

    # for i in range(bin_num):
    #     start_idx = remainder_ev_per_bin + i * nb_ev_per_bin
    #     end_idx = remainder_ev_per_bin + (i + 1) * nb_ev_per_bin
    #     hist[i, :] = sig_distr_filtered[args[start_idx:end_idx]]

    # create the signal hist; first bin has regular nb of events + remainder events
    # and is therefore a bit larger than the other bins
    hist = []
    hist.append(sig_distr_filtered[args[0:remainder_ev_per_bin+nb_ev_per_bin]])
    for i in range(1, bin_num):
        start_idx = remainder_ev_per_bin + i * nb_ev_per_bin
        end_idx = remainder_ev_per_bin + (i + 1) * nb_ev_per_bin
        hist.append(sig_distr_filtered[args[start_idx:end_idx]])

    if remainder_ev_per_bin == 0:
        odd_bin = None
        bins_limits = np.arange(bin_num) * nb_ev_per_bin # how many events in each bin
        bins_limits = args[bins_limits]
        bins_limits = np.concatenate(([hist_edge_l],sig_distr_filtered[bins_limits],
                                    [sig_distr_filtered[args[-1]]]))

    # else:
    #     # The odd bin is the first bin, which contains the remainder events
    #     odd_bin = sig_distr_filtered[args[:remainder_ev_per_bin]]

    #     # Die Limits starten bei 0 (für den odd_bin) und dann jeweils am Anfang der regulären Bins
    #     limit_indices = np.arange(bin_num) * nb_ev_per_bin + remainder_ev_per_bin
    #     limit_indices = np.concatenate(([0], limit_indices))

    #     bins_limits = sig_distr_filtered[args[limit_indices]]
    #     # Letztes Limit (das Maximum) wird wie gewohnt angehängt
    #     bins_limits = np.concatenate((bins_limits, [sig_distr_filtered[args[-1]]]))
    #     bins_limits[0]=hist_edge_l
    else:
        # write a binning where odd bin and first regular bin are both in first bin
        odd_bin = None
        limit_indices = np.arange(bin_num) * nb_ev_per_bin
        bins_limits = sig_distr_filtered[args[limit_indices]]
        bins_limits = np.concatenate(([hist_edge_l], bins_limits[1:], [sig_distr_filtered[args[-1]]]))


    return (hist, odd_bin, bins_limits)
