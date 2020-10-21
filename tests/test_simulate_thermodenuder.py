# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:55:59 2020

Tests for TD simulation

@author: J Kodros
"""

import pytest
import numpy as np

from src import thermodenuder
from src import massfractionremaining
from src.simulate_thermodenuder import fitTD


def get_norpinic_acid_sim(crt=True):
    if crt:
        f1 = np.load('tests/test_data/norpinicAcidFitting_CRT_2020.npz')
    else:
        f1 = np.load('tests/test_data/norpinicAcidFitting_mRes_2020.npz')
    cstar = f1['cstar']
    dh_vap = f1['dHvap']
    mfr_sim = f1['MFR_sim']
    mfr_obs = f1['MFR_obs']
    T_TD = f1['T_TD']
    f1.close()
    return cstar, dh_vap, mfr_sim, mfr_obs, T_TD


def get_norpinic_acid_inputs():
    td = thermodenuder.Thermodenuder(geometry='Patra')
    c_aer = 5.2 * 1e-9
    dp = 478. * 1e-9
    cstar, dh_vap, mfr_sim, mfr_obs, T_TD = get_norpinic_acid_sim()
    mfr = massfractionremaining.MassFractionRemaining(
        T_TD, mfr_obs, c_aer=c_aer, dp=dp)
    return mfr, td


def make_single_temp_input(T=25.0, mfr_obs=1.0, c_aer=5.2, dp=478,
                           cstar=10, dh=83):
    # Setup TD
    td = thermodenuder.Thermodenuder(geometry='Patra')
    # Setup MFR
    T_TD = np.array([T]) + 273.15
    MFR = np.array([mfr_obs])
    c_aer = c_aer * 1e-9
    dp = dp * 1e-9
    mfr = massfractionremaining.MassFractionRemaining(
        T_TD, MFR, c_aer=c_aer, dp=dp)
    # Setup guesses
    cstar = cstar * 1e-9
    dh = dh * 1e3
    return mfr, td, cstar, dh


class TestFitTD(object):
    def test_no_cooling_room_temp(self):
        mfr, td, cstar, dh = make_single_temp_input(
            T=25.0, mfr_obs=1.0, c_aer=5.2, dp=478., cstar=10., dh=83.)
        MFR_SIM = fitTD(mfr, td, cstar, dh, cooling_section=False)
        actual = MFR_SIM[0]
        expected = 1.0
        msg = ('No cooling section at 25 C returned {a}'.format(a=actual)
               + ' but expected {e}'.format(e=expected))
        assert actual == pytest.approx(expected), msg

    def test_norpinic_acid_crt(self):
        cstar, dh, mfr_sim_norpinic, mfr_obs, T_TD = get_norpinic_acid_sim()
        mfr, td = get_norpinic_acid_inputs()
        MFR_SIM = fitTD(mfr, td, cstar, dh, cooling_section=True)
        msg = 'Calculated CRT does not fit norpinic acid CRT'
        rtol = 1e-3
        atol = 1e-3
        assert np.allclose(MFR_SIM, mfr_sim_norpinic,
                           rtol=rtol, atol=atol), msg

    def test_norpinic_acid_mres(self):
        cstar, dh, mfr_sim_norpinic, mfr_obs, T_TD = get_norpinic_acid_sim()
        mfr, td = get_norpinic_acid_inputs()
        MFR_SIM = fitTD(mfr, td, cstar, dh, cooling_section=True, mres=True)
        msg = 'Calculated MRES does not fit norpinic acid CRT'
        rtol = 1e-3
        atol = 1e-3
        assert np.allclose(MFR_SIM, mfr_sim_norpinic,
                           rtol=rtol, atol=atol), msg