# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:42:20 2020

Class for fitting a thermogram to observations using the functions
in simulated_td

@author: J Kodros
"""
import numpy as np
from scipy.optimize import curve_fit

from src.simulate_thermodenuder import fitTD
from src.gridsearchtd import GridSearchTD


class ThermodenuderSimulator:
    def __init__(self, mfr, td, param_grid=None, cstar0=None, dh0=None,
                 fitting_method='curve_fit', cooling_section=True, mres=True):
        self.mfr = mfr
        self.td = td
        self.param_grid = param_grid
        self.cstar0 = cstar0
        self.dh0 = dh0
        self.fitting_method = fitting_method
        self.cooling_section = cooling_section
        self.mres = mres

    def add_param_grid(self, grid):
        self.param_grid = grid

    def fit(self):
        if self.fitting_method == 'curve_fit':
            self.fit_curve_fit()
        elif self.fitting_method == 'curve_fit_define_star':
            self.fit_curve_fit_define_cstar()
        elif (self.fitting_method == 'brute_force') or (
                self.fitting_method == 'grid_search'):
            self.fit_brute_force()
        else:
            print('Fitting option not implemented')

    def fit_brute_force(self):
        '''
        Fit simulated mfr to observations using a brute force (grid search)
        '''
        # Grid search parameters
        cstar = self.param_grid.get('cstar')
        dh = self.param_grid.get('dh')
        # Counter and out arrays
        ntrials = len(cstar) * len(dh)
        error = np.zeros((len(cstar), len(dh)))
        mfr_sims = np.zeros((len(self.mfr.T_TD), len(cstar), len(dh)))
        counter = 0
        for i in range(0, len(cstar)):
            for j in range(0, len(dh)):
                print('Trial number: '+str(counter+1)+' of '+str(ntrials))
                mfr_out = fitTD(self.mfr, self.td, cstar[i], dh[j],
                                cooling_section=self.cooling_section,
                                mres=self.mres)
                mfr_sims[:, i, j] = mfr_out
                error[i, j] = self.calc_error(mfr_out)
                counter += 1
        self.gs = GridSearchTD(self.mfr, self.td, mfr_sims, self.param_grid,
                               error=error)
        mfr_best, cstar_best, dh_best = self.gs.find_best_fit()
        self.error = error
        self.cstar_best = cstar_best
        self.dh_best = dh_best
        self.mfr_sim_best = mfr_best
        self.mfr_grid = mfr_sims

    def find_min_error_coords(self, error):
        idx_min = np.where(error == error.min())
        return idx_min

    def find_best_brute_force_params(self, error, mfr_sims, cstars, dhs):
        idx_min = self.find_min_error_coords(error)
        cstar_best = cstars[idx_min[0]][0]
        dh_best = dhs[idx_min[1]][0]
        return cstar_best, dh_best

    def fit_curve_fit(self):
        p0 = [self.cstar0, self.dh0]
        xdata = self.mfr.T_TD
        ydata = self.mfr.MFR
        popt, pcov = curve_fit(lambda x, a, b: fitTD(
            self.mfr, self.td, a, b, cooling_section=self.cooling_section,
            mres=self.mres), xdata, ydata, p0=p0)

        mfr_out = fitTD(
            self.mfr, self.td, popt[0], popt[1],
            cooling_section=self.cooling_section,
            mres=self.mres)

        error = self.calc_error(mfr_out)
        self.mfr_sim_best = mfr_out
        self.error = error
        self.cstar_best = popt[0]
        self.dh_best = popt[1]

    def fit_curve_fit_define_csar(self):
        p0 = [self.dh0]
        xdata = self.mfr.T_TD
        ydata = self.mfr.MFR
        popt, pcov = curve_fit(lambda x, b: fitTD(
            self.mfr, self.td, self.cstar0, b,
            cooling_section=self.cooling_section,
            mres=self.mres), xdata, ydata, p0=p0)

        mfr_out = fitTD(
            self.mfr, self.td, self.cstar0, popt[0],
            cooling_section=self.cooling_section,
            mres=self.mres)

        error = self.calc_error(mfr_out)
        self.mfr_sim_best = mfr_out
        self.error = error
        self.dh_best = popt[0]
        self.cstar_best = self.cstar0

    def calculate_error(self, mfr_out):
        diff = (self.mfr.MFR - mfr_out)**2
        error = (diff.sum())**(0.5)/len(self.mfr.MFR)
        return error

    def _check_attributes(self, attrs):
        for attr in attrs:
            try:
                attr
            except AttributeError:
                attr = None

    def write_out_npz(self, out_name):
        attrs = [self.gs, self.mfr_grid]
        self._check_attributes(attrs)
        np.savez(out_name+'.npz', cstar=self.cstar_best, dh=self.dh_best,
                 error=self.error, T_TD=self.mfr.T_TD, mfr_obs=self.mfr.MFR,
                 mfr_sim=self.mfr_sim_best, param_grid=self.param_grid,
                 mfr_grid=self.mfr_grid)
