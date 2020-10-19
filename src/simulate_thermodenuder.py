# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:39:08 2020

Simulate TD for single OA compounds

@author: J Kodros
"""

from pylab import *
from scipy import integrate
from src import fluxes as fx


def fitTD(T_TD, cstar, dHvap, alpha, dp_i, c_aer_tot_i, rho\
          , l_heat, D_heat, D_cool, l_cool, flow_ams, flow_smps\
          , MW=0.2, sigma=0.05, T_ref=298.15, T_i=298.15, T_cool=297.15):