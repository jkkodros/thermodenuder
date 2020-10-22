# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:42:55 2020

Test routines for the thermodenuder class and methods

@author: J Kodros
"""

import pytest
import numpy as np
from src import thermodenuder


def get_patra_td():
    flow_ams = 1.3E-6
    flow_smps = 1.6666666667E-5
    td = thermodenuder.Thermodenuder(geometry='Patra',
                                     flow_smps=flow_smps,
                                     flow_ams=flow_ams)
    return td


class TestCalcVolume(object):
    def test_with_ones(self):
        diameter = 2.0
        length = 1.0
        td = thermodenuder.Thermodenuder()
        vol = td.calc_volume(diameter, length)
        expected = 3.14
        msg = ('Calc volume returned {actual} instead of {expected}'.format(
            actual=vol, expected=expected))
        assert vol == pytest.approx(np.pi), msg


class TestCalcCenterlineResidenceTime(object):
    def test_centerline_with_ones(self):
        volume = 1.0
        flow_rate = 1.0
        td = thermodenuder.Thermodenuder()
        crt = td.calc_centerline_residence_time(volume, flow_rate)
        expected = 0.5
        msg = ('Calc crt returned {actual} instead of {expected}'.format(
            actual=crt, expected=expected))
        assert crt == pytest.approx(expected), msg

    def test_crt_patra(self):
        td = get_patra_td()
        volume = td.calc_volume(td.D_heat, td.l_heat)
        actual = td.calc_centerline_residence_time(volume, td.flow)
        expected = 14.4798861397275
        msg = ('Calculated Patra CRT {actual} instead of {expected}'.format(
            actual=actual, expected=expected))
        assert actual == pytest.approx(expected), msg


class TestCalcHeatingSectionCRT(object):
    def test_patra_heating_section(self):
        td = get_patra_td()
        actual = td.calc_heating_section_crt()
        expected = 14.4798861397275
        msg = ('Patra Heating CRT {actual} instead of {expected}'.format(
            actual=actual, expected=expected))
        assert actual == pytest.approx(expected), msg


class TestCalcCoolingSectionCRT(object):
    def test_patra_cooling_section(self):
        td = get_patra_td()
        actual = td.calc_cooling_section_crt()
        expected = 14.4798861397275
        msg = ('Patra Cooling CRT {actual} instead of {expected}'.format(
            actual=actual, expected=expected))
        assert actual == pytest.approx(expected), msg

    def test_patra_heat_is_equal_to_cool(self):
        td = get_patra_td()
        crt_heat = td.calc_heating_section_crt()
        crt_cool = td.calc_cooling_section_crt()
        actual = crt_heat - crt_cool
        expected = 0.0
        msg = ('Pata heating-cool CRT {a} instead of {e}'.format(
            a=actual, e=expected))
        assert actual == pytest.approx(expected), msg


class TestGetAvgResTimeRings(object):
    def test_patra_avg_res_time(self):
        td = get_patra_td()
        actual = td.get_avg_res_time_rings()
        expected = [
            14.552615030775144,
            14.85100525626096,
            15.48630657747207,
            16.548250838865382,
            18.213603842894386,
            20.83455866825633,
            25.183322100925412,
            33.2901551424942,
            52.66691730931345,
            153.1060918644558
            ]
        actual = np.array(actual)
        expected = np.array(expected)
        msg = ('Calculated rings does not match patra rings')
        assert np.allclose(actual, expected), msg


class TestGetRings(object):
    def test_patra_rings_area(self):
        expected = np.array([1.04062115e-05, 3.12186345e-05, 5.20310575e-05,
                             7.28434805e-05, 9.36559036e-05, 1.14468327e-04,
                             1.35280750e-04, 1.56093173e-04, 1.76905596e-04,
                             1.97718019e-04])
        td = get_patra_td()
        actual, area0 = td.get_rings()
        msg = ('Calculated rings Area did not match Patra Area')
        assert np.allclose(actual, expected), msg

    def test_patra_rings_area0(self):
        expected = 0.001040621150575083
        td = get_patra_td()
        Area, actual = td.get_rings()
        msg = ('Calculated Area0 {a} did not match patra {e}'.format(
            a=actual, e=expected))
        assert actual == pytest.approx(expected), msg
