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


def calc_number_from_mass(mfr):
    '''
    Calculate total number from total OA mass assuming density and
    spherical particles.
    '''
    n_tot = mfr.c_aer / (4.0 * mfr.density * mfr.rp**3 * np.pi/3.)
    return n_tot


def pstar_from_cstar(cstar, mfr, td, constants):
    '''
    Calculate saturation vapor concentration based off the given
    saturation mass concentration (cstar)
    '''
    pstar = (cstar * constants.IDEAL_GAS * td.T_ref) / mfr.MW
    return pstar


def calc_saturation_pressure_at_temperature(pstar, dh, td, constants):
    '''
    Calculate saturation vapor pressure with temperature dependence
    '''
    psat_i = pstar * np.exp(dh * ((1.0/td.T_ref) - (1.0/td.T_initial)) /
                            constants.IDEAL_GAS)
    return psat_i


def kelvin_effect_at_initial_temperature(mfr, td, constants):
    '''
    Calculate Kelvin effect at the initial TD temperature
    '''
    Ke_i = np.exp(2.0 * mfr.MW * mfr.sigma / (
        constants.IDEAL_GAS * td.T_initial * mfr.density * mfr.rp))
    return Ke_i


def equilibrium_pressure(psat, Ke):
    '''
    Calculate equilibrium pressure
    '''
    peq = psat * Ke
    return peq


def vapor_mass_concentration(peq, mfr, constants):
    '''
    Calculate the vapor concentration at the initial temperature
    '''
    cgas = peq * mfr.MW / (constants.IDEAL_GAS * mfr.T_TD[0])
    return cgas


def get_time_array(td, heating=True, mres=False, n=1000, t0=0):
    '''
    Get time and dt arrays for the integration
    '''
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


def calc_mass_in_rings(td, time_heat, Pc_t_k):
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
    '''
    Setup the scipy integrator
    '''
    r = integrate.ode(fx.calc_fluxes).set_integrator("dopri5")
    return r


def set_integrator_params(r, dt, length, crt, T, n_tot_i, pstar, dh, time, y0,
                          td, mfr, constants):
    '''
    Set the integrator parameters
    '''
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
    Note: This only does one section (not both heating and cooling)
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
        mp_out = calc_mass_in_rings(td, time, Pc_t_k)
    else:
        mp_out = Pc_t_k[-1]

    return mp_out


def fitTD(mfr, td, cstar, dh, cooling_section=True, mres=False):
    # Initialize object containing useful thermodynamic constants
    constants = ctd.ThermodynamicConstants()
    # Initial number concentration of OA
    n_tot_i = calc_number_from_mass(mfr)
    # Saturation vapor pressure (from given cstar)
    pstar = pstar_from_cstar(cstar, mfr, td, constants)
    # Saturation vapor pressure at initial temperature
    psat_i = calc_saturation_pressure_at_temperature(pstar, dh, td, constants)
    # Kevlin effect at initial temperature and radius
    Ke_i = kelvin_effect_at_initial_temperature(mfr, td, constants)
    # Equilibrium pressure above particle surface
    peq_i = equilibrium_pressure(psat_i, Ke_i)
    # Mass per particle
    mp_i = mfr.mass_per_particle()
    # Vapor mass concentration
    cgas_i = vapor_mass_concentration(peq_i, mfr, constants)
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
