import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats
import math


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def histHelper(n_bins, x_min, x_max, data, weights=0, where="mid", log=False):
    """
    Wrapper around the numpy histogram function.

    Arguments:
        n_bins {int} -- the number of bins
        x_min {float} -- the minimum number along x
        x_max {float} -- the maximum number along x
        data {2d tuple} -- array or list of arrays
        weigths {2d tuple} -- same shape as data or 0 if all weights are equal
        where {string} -- if where='post': duplicate the last bin
        log {bool} -- log==True: return x-axis log

    Outputs:
        edges {1d tuple} -- The bin edges
        edges_mid {1d tuple} -- The middle of the bins
        bins {2d tuple} -- The bin values, same depth as data
        max_val {1d tuple}-- the maximum value, same depth as data
    """
    if log:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
    else:
        edges = np.linspace(x_min, x_max, n_bins + 1)
    edges_mid = [edges[i] + (edges[i + 1] - edges[i]) / 2 for i in range(n_bins)]
    if weights == 0:
        weights = [[1] * len(d) for d in data]

    bins = [
        np.histogram(data_i, bins=edges, weights=weights_i)[0]
        for data_i, weights_i in zip(data, weights)
    ]
    max_val = [max(x) for x in bins]
    if where == "post":
        bins = [np.append(b, b[-1]) for b in bins]

    return edges, edges_mid, bins, max_val


def hist_bin_uncertainty(data, weights, x_min, x_max, bin_edges):
    """
    Calculate the error on the bins in the histogram including the weights.

    Arguments:
        edges {1d tuple} -- The bin edges
        data {1d tuple} -- array with the data of a variable
        weigths {1d tuple} -- weights, same shape as data
        
    Outputs:
        bin_uncertainties {1d tuple} -- Uncertainty on each bin
    """
    # Bound the data and weights to be within the bin edges
    mask_in_range = (data > x_min) & (data < x_max)
    in_range_data = data[mask_in_range]
    in_range_weights = weights[mask_in_range]

    # Bin the weights with the same binning as the data
    bin_index = np.digitize(in_range_data, bin_edges)
    # N.B.: range(1, bin_edges.size) is used instead of set(bin_index) as if
    # there is a gap in the data such that a bin is skipped no index would appear
    # for it in the set
    binned_weights = np.asarray(
        [
            in_range_weights[np.where(bin_index == idx)[0]]
            for idx in range(1, len(bin_edges))
        ]
    )
    bin_uncertainties = np.asarray(
        [np.sqrt(np.sum(np.square(w))) for w in binned_weights]
    )
    return bin_uncertainties


def kstest_weighted(data1, data2, wei1, wei2):
    """
    2-sample KS test unbinned probability.
    Takes into account the weight of the events.
    stackoverflow.com/questions/40044375/
    how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples/40059727
    
    Arguments:
        data1 {1d tuple} -- array with the data of a variable
        wei1 {1d tuple} -- weights, same shape as data
        data2 {1d tuple} -- array with the data of a variable
        wei2 {1d tuple} -- weights, same shape as data

    Outputs:
        d -- KS-text max separation
        prob -- KS-test p-value
    """
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = cwei1[tuple([np.searchsorted(data1, data, side="right")])]
    cdf2we = cwei2[tuple([np.searchsorted(data2, data, side="right")])]
    d = np.max(np.abs(cdf1we - cdf2we))
    # Note: d absolute not signed distance
    n1 = sum(wei1)
    n2 = sum(wei2)
    en = np.sqrt(n1 * n2 / float(n1 + n2))
    prob = scipy.stats.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    return d, prob


def get_ratio_ks(plot_data, weights):
    # KS-test
    flattened_MC = np.concatenate(np.array(plot_data)[[1, 3, 4]]).ravel()
    flattened_weights = np.concatenate(np.array(weights)[[1, 3, 4]]).ravel()
    ks_test_d, ks_test_p = kstest_weighted(
        flattened_MC, plot_data[2], flattened_weights, weights[2]
    )

    mc_weights = sum(np.concatenate(np.array(weights)[[1, 4]]).ravel())
    off_weights = sum(weights[3])
    on_weights = sum(weights[2])

    ratio1 = (on_weights - off_weights) / mc_weights
    ratio1_err = np.sqrt(mc_weights + off_weights) / mc_weights
    ratio2 = on_weights / (mc_weights + off_weights)
    ratio = [ratio1, ratio2, ratio1_err]
    return ratio, ks_test_p


class Plotter:
    """
    Class to make self.data/MC plots
    Initialise using a data dictionary as produced by neutrinoID_loader
    """

    # Fields shared of any class instance
    dict_names = ["nue", "nu", "on", "off", "dirt"]
    gr = (1 + 5 ** 0.5) / 2
    
    #colors:
    cat_c = {
    'cosmic': "xkcd:salmon",
    'outFV': "xkcd:brick",
    'inFV': "xkcd:green",
    'dirt': "xkcd:tomato",
    'off': "grey"
    }

    # Fields for initialisation
    def __init__(self, file_dict, width=3.5 * gr, height=4):
        self.data = file_dict
        self.width = width
        self.height = height
        print("Initialisation done")

    def binner(self, tree, field, x_min, x_max, n_bins, query):
        plot_data = []
        weights = []
        for k in self.dict_names:
            mask = self.data[k][tree][query] > 0
            plot_data.append(np.array(self.data[k][tree][field])[mask])
            weights.append(np.array(self.data[k][tree]["scale"])[mask])
        if tree=='slices':
            mask_cosmic = ((self.data["nu"][tree]["purity"]<0.5) | (self.data["nu"][tree]["completeness"]<0.5)) & (self.data["nu"][tree][query]>0)
        else:
            mask_cosmic = ( self.data["nu"][tree][query]<0 )
        fid_mask = (self.data["nu"][tree]["true_fidvol"]==0) & (self.data["nu"][tree][query]>0) & (~mask_cosmic)
        plot_data.append(np.array(self.data["nu"][tree][field][fid_mask]))
        weights.append(np.array(self.data["nu"][tree]["scale"][fid_mask]))
        plot_data.append(np.array(self.data["nu"][tree][field][mask_cosmic]))
        weights.append(np.array(self.data["nu"][tree]["scale"][mask_cosmic]))
        
        ratio, ks_p = get_ratio_ks(plot_data, weights)
        edges, edges_mid, bins, max_val = histHelper(
            n_bins, x_min, x_max, plot_data, weights=weights
        )
        bins_err = [
            hist_bin_uncertainty(d_i, w_i, x_min, x_max, edges)
            for d_i, w_i in zip(plot_data, weights)
        ]
        return plot_data, weights, ratio, ks_p, edges, edges_mid, bins, bins_err

    def data_mc_plot_add(
        self,
        tree,
        field,
        x_min,
        x_max,
        n_bins,
        x_lab="",
        y_lab="",
        title_left="",
        title_right="",
        query="run",  # default query selects all events
        ax=[],
    ):
        create_plot = len(ax)==0
        if create_plot:
            fig, ax = plt.subplots(
                ncols=1,
                nrows=2,
                figsize=(self.width, self.height),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex="col",
            )
        # Make the actual plot
        plot_data, weights, ratio, ks_p, edges, edges_mid, bins, bins_err = self.binner(
            tree, field, x_min, x_max, n_bins, query
        )
        widths = edges_mid - edges[:-1]
        # Data/MC
        ax[0].errorbar(
            edges_mid,
            bins[2],
            xerr=widths,
            yerr=bins_err[2],
            color="k",
            fmt="none",
            label="BNB On",
        )
        ax[0].bar(
            edges_mid, bins[3], lw=2, label="BNB Off", width=2 * widths, color=self.cat_c['off']
        )
        ax[0].bar(
            edges_mid,
            bins[1],
            lw=2,
            label=r"$\nu$ in FV (MC)",
            width=2 * widths,
            bottom=bins[3],
            color=self.cat_c['inFV'],
        )
        ax[0].bar(
            edges_mid,
            bins[5],
            lw=2,
            label=r"$\nu$ out FV (MC)",
            width=2 * widths,
            bottom=bins[3] + bins[1] - bins[5],
            color=self.cat_c['outFV'],
        )
        if tree=='slices':
            ax[0].bar(
                edges_mid,
                bins[6],
                lw=2,
                label=r"Cosmic (MC)",
                width=2 * widths,
                bottom=bins[3] + bins[1] - bins[5] - bins[6],
                color=self.cat_c['cosmic'],
            )
            
        ax[0].bar(
            edges_mid,
            bins[4],
            lw=2,
            label=r"$\nu$ dirt (MC)",
            width=2 * widths,
            bottom=bins[1] + bins[3],
            color=self.cat_c['dirt'],
        )

        y_err = np.sqrt(bins_err[1] ** 2 + bins_err[3] ** 2 + bins_err[4] ** 2)
        val = bins[3] + bins[1] + bins[4]
        for m, v, e, w in zip(edges_mid, val, y_err, widths):
            ax[0].add_patch(
                patches.Rectangle(
                    (m - w, v - e),
                    2 * w,
                    2 * e,
                    hatch="\\\\\\\\\\",
                    Fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )
            sc_err = e / v
            ax[1].add_patch(
                patches.Rectangle(
                    (m - w, 1 - sc_err),
                    2 * w,
                    sc_err * 2,
                    hatch="\\\\\\\\\\",
                    Fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )
        ax[1].errorbar(
            edges_mid,
            bins[2] / val,
            xerr=widths,
            yerr=bins_err[2] / val,
            alpha=1.0,
            color="k",
            fmt="none",
        )
        yr_min = max(ax[1].get_ylim()[0],0)
        yr_max = min(ax[1].get_ylim()[1],2)
        ax[1].set_ylim(yr_min, yr_max)
        ax[0].set_title(title_right, loc="right")
        ax[1].set_xlabel(x_lab)
        ax[1].set_xlim(x_min, x_max)
        ax[1].set_ylabel(r"$\frac{Beam\ ON}{Beam\ OFF + MC}$")

        if y_lab == "":
            y_lab = tree.capitalize() + " per bin"
        ax[0].set_ylabel(y_lab)
        if title_left == "":
            ax[0].set_title(
                "(On-Off)/MC:{0:.2f}$\pm${1:.2f}".format(
                    ratio[0], round_up(ratio[2], 2)
                ),
                loc="left",
            )
        else:
            ax[0].set_title(title_left, loc="left")
        if create_plot:
            return fig, ax, ks_p, ratio
        else:
            return ks_p, ratio

    def data_mc_plot_sub(
        self,
        tree,
        field,
        x_min,
        x_max,
        n_bins,
        x_lab="",
        y_lab="",
        title_left="",
        title_right="",
        query="run",  # default query selects all events
        ax=[],
    ):
        create_plot = len(ax)==0
        if create_plot:
            fig, ax = plt.subplots(
                ncols=1,
                nrows=2,
                figsize=(self.width, self.height),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex="col",
            )
        # Make the actual plot
        plot_data, weights, ratio, ks_p, edges, edges_mid, bins, bins_err = self.binner(
            tree, field, x_min, x_max, n_bins, query
        )
        widths = edges_mid - edges[:-1]
        # Data/MC
        ax[0].errorbar(
            edges_mid,
            bins[2] - bins[3],
            xerr=widths,
            yerr=np.sqrt(bins_err[2] ** 2 + bins_err[3] ** 2),
            color="k",
            fmt="none",
            label="BNB On- BNB Off",
        )
        ax[0].bar(
            edges_mid,
            bins[1],
            lw=2,
            label=r"$\nu$ in FV (MC)",
            width=2 * widths,
            color=self.cat_c['inFV'],
        )
        ax[0].bar(
            edges_mid,
            bins[5],
            lw=2,
            label=r"$\nu$ out FV (MC)",
            width=2 * widths,
            bottom=bins[1] - bins[5],
            color=self.cat_c['outFV'],
        )
        if tree=='slices':
            ax[0].bar(
                edges_mid,
                bins[6],
                lw=2,
                label=r"Cosmic (MC)",
                width=2 * widths,
                bottom=bins[1] - bins[5] - bins[6],
                color=self.cat_c['cosmic'],
            )
            
        ax[0].bar(
            edges_mid,
            bins[4],
            lw=2,
            label=r"$\nu$ dirt (MC)",
            width=2 * widths,
            bottom=bins[1],
            color=self.cat_c['dirt'],
        )

        y_err = np.sqrt(bins_err[1] ** 2 + bins_err[4] ** 2)
        val = bins[1] + bins[4]
        for m, v, e, w in zip(edges_mid, val, y_err, widths):
            ax[0].add_patch(
                patches.Rectangle(
                    (m - w, v - e),
                    2 * w,
                    2 * e,
                    hatch="\\\\\\\\\\",
                    Fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )
            sc_err = e / v
            ax[1].add_patch(
                patches.Rectangle(
                    (m - w, 1 - sc_err),
                    2 * w,
                    sc_err * 2,
                    hatch="\\\\\\\\\\",
                    Fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )
        ax[1].errorbar(
            edges_mid,
            (bins[2] - bins[3]) / val,
            xerr=widths,
            yerr=np.sqrt(bins_err[2] ** 2 + bins_err[3] ** 2) / val,
            alpha=1.0,
            color="k",
            fmt="none",
        )
        yr_min = max(ax[1].get_ylim()[0],0)
        yr_max = min(ax[1].get_ylim()[1],2)
        ax[1].set_ylim(yr_min, yr_max)
        ax[0].set_title(title_right, loc="right")
        ax[1].set_xlabel(x_lab)
        ax[1].set_xlim(x_min, x_max)
        ax[1].set_ylabel(r"$\frac{Beam\ ON - OFF}{MC}$")

        if y_lab == "":
            y_lab = tree.capitalize() + " per bin"
        ax[0].set_ylabel(y_lab)
        if title_left == "":
            ax[0].set_title(
                "(On-Off)/MC:{0:.2f}$\pm${1:.2f}".format(
                    ratio[0], round_up(ratio[2], 2)
                ),
                loc="left",
            )
        else:
            ax[0].set_title(title_left, loc="left")
        if create_plot:
            return fig, ax, ks_p, ratio
        else:
            return ks_p, ratio

    def data_mc_plot_both(
        self,
        tree,
        field,
        x_min,
        x_max,
        n_bins,
        x_lab="",
        y_lab="",
        title_left="",
        title_right="",
        query="run",  # default query selects all events
    ):
        ax1, fig1, ks_p, ratio = self.data_mc_plot_add(
            tree,
            field,
            x_min,
            x_max,
            n_bins,
            x_lab,
            y_lab,
            title_left,
            title_right,
            query,
        )
        ax2, fig2, _, _ = self.data_mc_plot_sub(
            tree,
            field,
            x_min,
            x_max,
            n_bins,
            x_lab,
            y_lab,
            title_left,
            title_right,
            query,
        )
        return [ax1, ax2], [fig1, fig2], ks_p, ratio
    
    
def efficiency(
    num, den, num_w=None, den_w=None, n_bins=10, x_min=0, x_max=10, conf_level=None
):
    """
    Calculate the efficiency given two populations: one containg 
    the totatility of the events,and one containing only events 
    that pass the selection.
    It uses a frequentist approach to evaluate the uncertainty.
    Other methods are to be implemented.
    
     Arguments:
        num {tuple} -- The totality of the events
        den {tuple} -- The events that pass the selection
        num_w {tuple} -- Optional, the weight for every event
        den_w {tuple} -- Optional, the weight for every selected event
        n_bins {int} -- Optional, the number of bins
        x_min {float} -- Optional, the minimum number along x
        x_max {float} -- Optional, the maximum number along x
        conf_level {float} -- Optional, the confidence level to be used
        
    Outputs:
        eff {tuple} -- The efficiency per bin
        unc_low {tuple} -- The lower uncertainty per bin
        unc_up {tuple} -- The upper uncertainty per bi
        bins {tuple} -- The bin edges
    """

    if num_w is None:
        num_w = [1.0] * len(num)

    if den_w is None:
        den_w = [1.0] * len(den)

    if conf_level is None:
        conf_level = 0.682689492137

    num = np.asarray(num, dtype=np.float32)
    num_w = np.asarray(num_w, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    den_w = np.asarray(den_w, dtype=np.float32)

    bins = np.linspace(x_min, x_max, n_bins)

    num_h, _ = np.histogram(num, bins=bins)
    num_w_h, _ = np.histogram(num, weights=num_w, bins=bins)
    num_w2_h, _ = np.histogram(num, weights=num_w ** 2, bins=bins)

    den_h, _ = np.histogram(den, bins=bins)
    den_w_h, _ = np.histogram(den, weights=den_w, bins=bins)
    den_w2_h, _ = np.histogram(den, weights=den_w ** 2, bins=bins)

    eff = num_w_h / den_w_h

    variance = (num_w2_h * (1.0 - 2 * eff) + den_w2_h * eff * eff) / (den_w_h * den_w_h)
    sigma = np.sqrt(variance)
    prob = 0.5 * (1.0 - conf_level)
    delta = -scipy.stats.norm.ppf(prob) * sigma

    unc_up = []
    unc_low = []

    for eff_i, delta_i in zip(eff, delta):
        if eff_i - delta_i < 0:
            unc_low.append(eff_i)
        else:
            unc_low.append(delta_i)

        if eff_i + delta_i > 1:
            unc_up.append(1.0 - eff_i)
        else:
            unc_up.append(delta_i)

    return eff, unc_low, unc_up, bins


# Helper class duplicating the last bin, useful to use in combination with matplotlib step function.
def efficiency_post(
    num, den, num_w=None, den_w=None, n_bins=10, x_min=0, x_max=10, conf_level=None
):
    eff, unc_low, unc_up, edges = efficiency(
        num, den, num_w, den_w, n_bins, x_min, x_max, conf_level
    )
    eff = np.append(eff, eff[-1])
    unc_low = np.append(unc_low, unc_low[-1])
    unc_up = np.append(unc_up, unc_up[-1])
    return eff, unc_low, unc_up, edges
