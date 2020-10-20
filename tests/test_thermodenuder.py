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
