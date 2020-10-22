# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:35:09

Class to contain MFR observations and aerosol properties as inputs to
TD simulation

@author: J Kodros
"""
import numpy as np
import matplotlib.pyplot as plt

class MassFractionRemaining:
    def __init__(self, T_TD, MFR, c_aer=None, dp=None, density=1300,
                 MW=0.17, sigma=0.05, alpha=1.0):
        '''
        Units are assumed to be SI.
        T in [K], c_aer in [kg m-3], dp in [m], density in [kg m-3]
        MW in [kg mol-1] and sigma in [N m-1]
        '''
        self.T_TD = T_TD
        self.MFR = MFR
        self.c_aer = c_aer
        self.dp = dp
        self.rp = dp/2.0
        self.density = density
        self.MW = MW
        self.sigma = sigma
        self.alpha = alpha

    def mass_per_particle(self):
        mass_per_particle = self.density * 4.0 * np.pi * self.rp**3 / 3.0
        return mass_per_particle

    def plot_mfr(self, ax=None, celsius=True, xlim=[30, 100], ylim=[0, 1],
                 xlabel='Temperature', ylabel='MFR'):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if celsius:
            temperature = self.T_TD - 273.15
        else:
            temperature = self.T_TD
        ax.plot(temperature, self.MFR, marker='o', linestyle='none', color='r')
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        return ax