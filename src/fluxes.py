# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:44:43 2019

Fluxes size dist

@author: J Kodros
"""

from pylab import *

def calc_fluxes(t, y, dt, l_heat, t_res_heat, T_f, T_i, n_tot0, pstar\
                , dHvap, T_ref, MW, sigma, rho, Dn, mu, press, alpha, R):


    Pc = y[0]
    Gc = y[1]

    #xcoord = l_heat*t/t_res_heat
    T_TDc = T_f

    #xcoord0 = l_heat*(t-dt)/t_res_heat
    #T_TD0 = T_f

    # Total aerosol number concentation
    n_tot = n_tot0 * T_i/T_TDc

    # Check for negatives and/or nans
    Pc = ma.masked_less(Pc, 0).filled(0.0)
    Pc = ma.masked_invalid(Pc).filled(0.0)
    Gc = ma.masked_less(Gc, 0).filled(0.0)
    Gc = ma.masked_invalid(Gc).filled(0.0)

    # Equilibrium pressures at the TD temperature
    psat = pstar * exp(dHvap * ((1.0/T_ref) - (1.0/T_TDc))/R)
    csat = MW * psat / (R * T_TDc)

    # Total particle mass
    mp = Pc

    # Diffusion coefficients of the species
    D = Dn * (T_TDc/T_ref)**mu

    #### Calculating the composition
    if mp > 1.0E-22:
        #n_apu = 1.0/MW

        # Particle volume and size
        vp = mp/rho
        rp = (3.0 * vp / (4.0*pi))**(1.0/3.0)

        # Kelvin effect
        Ke = exp(2.0 * MW * sigma / (R*T_TDc*rho*rp))

        # Equilibrium pressures
        peq = psat * Ke

        ### Transitional correction
        # Mean velocity of the gas molecules
        c_avg = (8.0 * R * T_TDc / (MW * pi)) ** (0.5)

        # mean free path of the gas molecules
        lambda_mfp = 3.0 * D / c_avg

        # Knudsen number
        Kn = lambda_mfp / rp

        # Fucks and Sutugin transition regime correction
        beta = (1.0 + Kn) / (1.0 + (4.0/(3.0*alpha) + 0.377)*Kn + 4*Kn**2/(3.0*alpha))

    else:
        mp = 0.0
        vp = 0.0
        rp = 0.0
        peq = 0.0
        beta = 0.0

    # pressure at the particle surface
    pv_a = peq

    # partial pressure far away from the particles
    pv_i = Gc * R * T_TDc / MW

    if mp > 0.0 and (1.0 - pv_a/press)/(1.0 - pv_i/press) > 0.0:
        flx = 4.0 * pi * rp * press *D * beta * MW * log((1.0 - pv_a/press)/(1.0 - pv_i/press))/(R*T_TDc)
    elif mp > 0.0 and (1.0 - pv_a/press)/(1.0 - pv_i/press) <= 0.0:
        flx = 4.0 * pi * rp * D * beta * MW * (pv_i-pv_a)/(R*T_TDc)
    else:
        flx = 0.0

    # mass flux of each compound to the gas phase
    flx2 = -1 * n_tot * flx

    # Check for bad values
    output = array([flx, flx2])

    return output


