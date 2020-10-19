# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:40:46 2020

Thermodenuder

@author: J Kodros
"""
import numpy as np


class Thermodenuder:
    def __init__(self, D_heat=0.5, l_heat=0.5, D_cool=0.0364, l_cool=0.0364,
                 flow_ams=1.3E-6, flow_smps=1.67E-5, T_initial=298.15,
                 T_cool=297.15, geometry=None):
        # Geometry presets
        if geometry == 'Patra':
            self.D_heat = 0.0364    # [m]
            self.l_heat = 0.5     # [m]
            self.D_cool = 0.0364    # [m]
            self.l_col = 0.5    # [m]
        else:
            self.D_heat = D_heat
            self.l_heat = l_heat
            self.D_cool = D_cool
            self.l_cool = l_cool
        # Non-geomtry variables
        self.T_initial = T_initial
        self.T_cool = T_cool
        self.flow = flow_smps + flow_ams

    def calc_volume(self, diameter, length):
        volume = np.pi * (diameter/2.)**2 * length
        return volume

    def calc_centerline_residence_time(self, volume, flow_rate):
        crt = 0.5 * volume / flow_rate
        return crt
