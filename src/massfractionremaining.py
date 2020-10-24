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
        MassFractionRemaining contains methods and attributes related to the
        observed aerosol proporties and observed MFR at each temperature.

        Attributes
        ----------
        T_TD : Numpy array
            Temperature [K] for thermodenunder
        MFR : Numpy array
            Observed mass fraction remaining (MFR) at each temperatue [ratio]
        c_aer : float, optional
            Average organic aerosol mass concentration [kg m-3].
            The default is None.
        dp : floar, optional
            Average volume mode diameter [m]. The default is None.
        density : float, optional
            OA density [kg m-3]. The default is 1300.
        MW : float, optional
            Molar mass [kg mol-1]. The default is 0.17.
        sigma : float, optional
            Surface tesion [N m-1]. The default is 0.05.
        alpha : float, optional
            Accomodation coefficient [unitless]. The default is 1.0.
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

    def calculate_mass_per_particle(self):
        '''
        Calculate the mass per particle assuming spherical particle with
        given density.

        Returns
        -------
        mass_per_particle : FLOAT
            The mass per particle with given density and radius

        '''
        mass_per_particle = self.density * 4.0 * np.pi * self.rp**3 / 3.0
        return mass_per_particle

    def calculate_number_from_mass(self):
        '''
        Calculate particle number concentration from given mass and radius.
        Assumes monodisperse population.

        Returns
        -------
        n_tot : float
            Number concentration [m-3] of OA given density and radius

        '''
        n_tot = self.c_aer / (4.0 * self.density * self.rp**3 * np.pi/3.)
        return n_tot

    def plot_mfr(self, ax=None, celsius=True, xlim=[30, 100], ylim=[0, 1],
                 xlabel='Temperature', ylabel='MFR'):
        '''Plot the observed mfr.'''
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if celsius:
            temperature = self.T_TD - 273.15
        else:
            temperature = self.T_TD
        ax.plot(temperature, self.MFR, marker='o', linestyle='none', color='r')
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        return ax
