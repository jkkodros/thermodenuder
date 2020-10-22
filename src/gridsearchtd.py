# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:33:16 2020

Class to handle the grid search/brute force results for TD fitting

@author: J Kodros
"""
import numpy as np
import matplotlib.pyplot as plt


class GridSearchTD:
    def __init__(self, mfr, td, mfr_sims, param_grid, error=None):
        self.mfr = mfr
        self.td = td
        self.mfr_sims = mfr_sims
        self.param_grid = param_grid
        self.error = error

    def find_min_error_coords(self):
        idx_min = np.where(self.error == self.error.min())
        return idx_min

    def find_best_fit(self):
        idx_min = self.find_min_error_coords()
        cstar = self.param_grid.get('cstar')
        dh = self.param_grid.get('dh')
        cstar_best = cstar[idx_min[0]][0]
        dh_best = dh[idx_min[1]][0]
        mfr_sim_best = self.mfr_sims[:, idx_min[0], idx_min[1]]
        return cstar_best, dh_best, mfr_sim_best

    def select_error_threshold_from_percentile(self, perc=0.95):
        error_flat = self.error.flatten()
        error_sorted = np.sort(error_flat)[::-1]
        x = np.linspace(0, 100, len(error_sorted))
        ix = np.abs(x - perc).argmin()
        threshold = error_sorted[ix]
        return threshold

    def get_mfr_sims_below_error_threshold(self, threshold):
        idx_mins = np.where(self.error <= threshold)
        mfr_uncertainty = self.mfr_sims[:, idx_mins[0], idx_mins[1]]
        return mfr_uncertainty

    def mfr_limits(self, mfr_uncertainty):
        mfr_lower = mfr_uncertainty.min(axis=1)
        mfr_upper = mfr_uncertainty.max(axis=1)
        return mfr_lower, mfr_upper

    def get_mfr_range(self, perc=0.95):
        threshold = self.select_error_threshold_from_percentile(perc=perc)
        mfr_uncertainty = self.get_mfr_sims_below_error_threshold(threshold)
        mfr_lower, mfr_upper = self.mfr_limits(mfr_uncertainty)
        return mfr_lower, mfr_upper

    def get_parameters_below_threshold(self, threshold):
        idx_mins = np.where(self.error <= threshold)
        cstar = self.param_grid('cstar')
        dh = self.param_grid('dh')
        cstar_uncertainty = cstar[idx_mins[0]]
        dh_uncertainty = dh[idx_mins[1]]
        return cstar_uncertainty, dh_uncertainty

    def plot_mfr_range(self, perc=0.95, ax=None, celsius=True):
        c, h, mfr_best = self.find_best_fit()
        mfr_lower, mfr_upper = self.get_mfr_range(perc=perc)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if celsius:
            temperature = self.mfr.T_TD - 273.15
        else:
            temperature = self.mfr.T_TD
        ax.plot(temperature, mfr_best, color='k', linewidth=3)
        ax.fill_between(temperature, mfr_lower, mfr_upper, color='grey',
                        alpha=0.4)
        return ax