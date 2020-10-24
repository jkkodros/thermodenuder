# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:39:08 2020

Simulate TD for single OA compounds

@author: J Kodros
"""

import numpy as np
from scipy import integrate

from src import fluxes as fx
from src import constants as ctd
from src import thermodynamic_equations as eqns


def calculate_initial_vapor_concentration(pstar, dh, mfr, td, constants):
    '''Calculate the initial vapor concentration at TD entrance.'''
    # Saturation vapor pressure at initial temperature
    psat_i = eqns.calculate_saturation_pressure_at_temperature(
        pstar, dh, td.T_ref, td.T_initial, constants.IDEAL_GAS)
    # Kevlin effect at initial temperature and radius
    Ke_i = eqns.calculate_kelvin_effect(
        mfr.MW, mfr.sigma, td.T_initial, mfr.density, mfr.rp,
        constants.IDEAL_GAS)
    # Equilibrium pressure above particle surface
    peq_i = eqns.calculate_equilibrium_pressure(psat_i, Ke_i)
    # Vapor mass concentration at initial temperature
    cgas_i = eqns.calculate_vapor_mass_concentration(
        peq_i, mfr.MW, td.T_initial, constants.IDEAL_GAS)
    return cgas_i


def get_time_array(td, heating=True, mres=False, n=1000, t0=0):
    '''
    Get time and dt arrays for the integration.'''
    if heating:
        if mres:
            avg_res = td.get_avg_res_time_rings()
            crt = avg_res[-1]
        else:
            crt = td.calc_heating_section_crt()
    else:
        crt = td.calc_cooling_section_crt()
    time = np.linspace(t0, crt, n)
    dt = np.mean(time[1:] - time[:-1])
    return time, dt


def calculate_mass_in_rings(td, time_heat, Pc_t_k):
    '''Calcute mass in each ring of TD for multiple residence time model.'''
    Area, Area0 = td.get_rings()
    avg_res = td.get_avg_res_time_rings()
    # multiple residence time
    mass = 0.0
    for i in range(0, len(Area)):
        if i == 0:
            idx = 0
        else:
            idx = np.abs(time_heat - avg_res[i]).argmin()
        mass = mass + Pc_t_k[idx]*(Area[i]/Area0)
    return mass


def initiate_integrator():
    '''Setup the scipy integrator.'''
    r = integrate.ode(fx.calc_fluxes).set_integrator("dopri5")
    return r


def set_integrator_params(r, dt, length, crt, T, n_tot_i, pstar, dh, time, y0,
                          td, mfr, constants):
    '''Set the integrator parameters.'''
    r.set_f_params(dt, length, crt, T, td.T_initial, n_tot_i, pstar, dh,
                   td.T_ref,
                   mfr.MW, mfr.sigma, mfr.density, constants.DIFFUSION_COEF,
                   constants.MU, constants.PRESSURE, mfr.alpha,
                   constants.IDEAL_GAS)
    r.set_initial_value(y0, time[0])
    return r


def simulate_one_temperature(mp_in, gc_in, T, r, dt, time, length, crt,
                             n_tot_i, pstar, dh, td, mfr, constants,
                             mres=False):
    '''
    Simulate the TD at a given temperature.
    Returns the estimate aerosol mass concentration at the end of the section
    Note: This only does one section (not both heating and cooling).
    '''
    y0 = np.array([mp_in, gc_in])
    y = np.zeros((len(time), len(y0)))
    y[0, :] = y0

    r = set_integrator_params(r, dt, length, crt, T, n_tot_i, pstar, dh,
                              time, y0, td, mfr, constants)

    for i in range(1, time.size):
        y[i, :] = r.integrate(time[i])
        if not r.successful():
            raise RuntimeError("Could not integrate")

    Pc_t_k = y[:, 0]
    Gc_t_k = y[:, 1]

    # Check for bad values
    Pc_t_k = np.ma.masked_less(Pc_t_k, 0.0).filled(0.0)
    Gc_t_k = np.ma.masked_less(Gc_t_k, 0.0).filled(0.0)

    if mres:
        mp_out = calculate_mass_in_rings(td, time, Pc_t_k)
    else:
        mp_out = Pc_t_k[-1]

    return mp_out


def fitTD(mfr, td, cstar, dh, cooling_section=True, mres=False):
    # Initialize object containing useful thermodynamic constants
    constants = ctd.ThermodynamicConstants()
    # Calculate total number concentration and mass per particle at entrance
    n_tot_i = mfr.calculate_number_from_mass()
    mp_i = mfr.calculate_mass_per_particle()
    # Saturation vapor pressure (from given cstar)
    pstar = eqns.convert_saturation_concentration_to_pressure(
        cstar, mfr.MW, td.T_ref, constants.IDEAL_GAS)
    # Calculate the initial vapor concentration at entrance
    cgas_i = calculate_initial_vapor_concentration(
        pstar, dh, mfr, td, constants)
    # Get time arrays
    time_heat, dt_heat = get_time_array(td, heating=True, mres=mres)
    time_cool, dt_cool = get_time_array(td, heating=False)
    # Output MFR array
    MFR_SIM = np.zeros(len(mfr.T_TD))
    # Initiate integrator
    r = initiate_integrator()

    # Loop through temperatures for heating section
    crt_heat = td.calc_heating_section_crt()
    crt_cool = td.calc_cooling_section_crt()
    for k, T in enumerate(mfr.T_TD):
        mp_f = simulate_one_temperature(
            mp_i, cgas_i, T, r, dt_heat, time_heat, td.l_heat, crt_heat,
            n_tot_i, pstar, dh, td, mfr, constants, mres=mres)

        if cooling_section:
            mp_f = simulate_one_temperature(
                mp_f, 0.0, td.T_cool, r, dt_cool, time_cool, td.l_cool,
                crt_cool, n_tot_i, pstar, dh, td, mfr, constants)

        c_aer_out = n_tot_i * mp_f

        MFR_SIM[k] = c_aer_out / mfr.c_aer

    return MFR_SIM
