# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:44:43 2019

Fluxes size dist

@author:
"""
import numpy as np


def calc_fluxes(t, y, dt, length, t_res, T_f, T_i, n_tot0, pstar,
                dh_vap, T_ref, MW, sigma, rho, Dn, mu, press, alpha, R):
    '''
    This is the main function for the ode to calculate particle flux
    through the TD. It is largely unchanged from the original model.
    '''
    # Particle and gas concentration
    Pc = y[0]
    Gc = y[1]
    # Total aerosol number concentation
    n_tot = n_tot0 * T_i/T_f
    # Check for negatives and/or nans
    Pc = np.ma.masked_less(Pc, 0).filled(0.0)
    Pc = np.ma.masked_invalid(Pc).filled(0.0)
    Gc = np.ma.masked_less(Gc, 0).filled(0.0)
    Gc = np.ma.masked_invalid(Gc).filled(0.0)
    # Equilibrium pressures at the TD temperature
    psat = pstar * np.exp(dh_vap * ((1.0/T_ref) - (1.0/T_f))/R)
    # Diffusion coefficients of the species
    D = Dn * (T_f/T_ref)**mu

    # Calculating the composition
    if Pc > 1.0E-22:
        # Particle volume and size
        vp = Pc/rho
        rp = (3.0 * vp / (4.0*np.pi))**(1.0/3.0)
        # Kelvin effect
        Ke = np.exp(2.0 * MW * sigma / (R*T_f*rho*rp))
        # Equilibrium pressures
        peq = psat * Ke
        # Transitional correction
        # Mean velocity of the gas molecules
        c_avg = (8.0 * R * T_f / (MW * np.pi)) ** (0.5)
        # mean free path of the gas molecules
        lambda_mfp = 3.0 * D / c_avg
        # Knudsen number
        Kn = lambda_mfp / rp
        # Fucks and Sutugin transition regime correction
        beta = ((1.0 + Kn) /
                (1.0 + (4.0/(3.0*alpha) + 0.377)*Kn + 4*Kn**2/(3.0*alpha)))
    else:
        Pc = 0.0
        vp = 0.0
        rp = 0.0
        peq = 0.0
        beta = 0.0

    # pressure at the particle surface
    pv_a = peq
    # partial pressure far away from the particles
    pv_i = Gc * R * T_f / MW
    if Pc > 0.0 and (1.0 - pv_a/press)/(1.0 - pv_i/press) > 0.0:
        flx = (4.0 * np.pi * rp * press * D * beta * MW *
               np.log((1.0 - pv_a/press)/(1.0 - pv_i/press))/(R*T_f))
    elif Pc > 0.0 and (1.0 - pv_a/press)/(1.0 - pv_i/press) <= 0.0:
        flx = 4.0 * np.pi * rp * D * beta * MW * (pv_i-pv_a)/(R*T_f)
    else:
        flx = 0.0

    # mass flux of each compound to the gas phase
    flx2 = -1 * n_tot * flx

    # Check for bad values
    output = np.array([flx, flx2])

    return output
