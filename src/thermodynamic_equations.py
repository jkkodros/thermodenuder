# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:00:41 2020

@author: LAQS
"""
import numpy as np


def convert_saturation_concentration_to_pressure(cstar, mw, temperature,
                                                 ideal_gas_const):
    '''
    Convert given saturation mass concentration into saturation
    vapor pressure.

    Parameters
    ----------
    cstar : float
        Saturation mass concentration [kg m-3].
    mw : Molar mass [kg mol-1]
        Molar mass of aerosol.
    temperature: float
        Temperature of plume [K].
    ideal_gas_const : float
        Ideal gas constant.

    Returns
    -------
    pstar : float
        Saturation vapor pressure [Pa].

    '''
    pstar = (cstar * ideal_gas_const * temperature) / mw
    return pstar


def calculate_saturation_pressure_at_temperature(
        pstar, dh, temperature_ref, temperature_new, ideal_gas_const):
    '''
    Calculate saturation vapor pressure with temperature dependence.

    Parameters
    ----------
    pstar : float
        Saturation vapor pressure [Pa].
    dh : float
        Enthalpy [J mol-1].
    temperature_ref : float
        Reference temperature [K].
    temperature_new: float
        New temperature [K].
    ideal_gas_const : float
        Ideal gas constant.

    Returns
    -------
    psat : float
        Saturation vapor pressure at new temperature [K].

    '''
    psat = pstar * np.exp(dh * ((1.0/temperature_ref) -
                                (1.0/temperature_new)) / ideal_gas_const)
    return psat


def calculate_kelvin_effect(mw, sigma, temperature, density, rp,
                            ideal_gas_const):
    '''
    Calculate Kelvin effect at temperature.

    Parameters
    ----------
    mw : float
        Molar mass [kg mol-1] of particle
    sigma : float
        Surface tension of particle [N m-1]
    temperature : float
        Temperature of plume [K]
    density : float
        Particle density [kg m-3]
    rp : float
        Radius of particle [m]
    ideal_gas_const : float
        Ideal gas constant

    Returns
    -------
    Ke : float
        Kelvin effect

    '''
    Ke = np.exp(2.0 * mw * sigma /
                (ideal_gas_const * temperature * density * rp))
    return Ke


def calculate_equilibrium_pressure(psat, Ke):
    '''Calculate equilibrium pressure above particle surface.'''
    peq = psat * Ke
    return peq


def calculate_vapor_mass_concentration(peq, mw, temperature, ideal_gas_const):
    '''
    Calculate the equilibrium vapor mass concentration above particle.

    Parameters
    ----------
    peq : float
        Equibirum pressure.
    mw : float
        Molar mass [kg mol-1].
    temperature : float
        Temperature of plume [K].
    ideal_gas_const : float
        Ideal gas constant.

    Returns
    -------
    cgas : float
        Vapor mass concentration at particle surface.

    '''
    cgas = peq * mw / (ideal_gas_const * temperature)
    return cgas
