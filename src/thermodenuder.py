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
                 T_cool=297.15, T_ref=298.15, geometry=None):
        # Geometry presets
        if geometry == 'Patra':
            self.D_heat = 0.0364    # [m]
            self.l_heat = 0.5     # [m]
            self.D_cool = 0.0364    # [m]
            self.l_cool = 0.5    # [m]
        else:
            self.D_heat = D_heat
            self.l_heat = l_heat
            self.D_cool = D_cool
            self.l_cool = l_cool
        # Non-geomtry variables
        self.T_initial = T_initial
        self.T_cool = T_cool
        self.T_ref = T_ref
        self.flow = flow_smps + flow_ams

    def calc_volume(self, diameter, length):
        volume = np.pi * (diameter/2.)**2 * length
        return volume

    def calc_centerline_residence_time(self, volume, flow_rate):
        crt = 0.5 * volume / flow_rate
        return crt

    def calc_heating_section_crt(self):
        volume = self.calc_volume(self.D_heat, self.l_heat)
        crt = self.calc_centerline_residence_time(volume, self.flow)
        return crt

    def calc_cooling_section_crt(self):
        volume = self.calc_volume(self.D_cool, self.l_cool)
        crt = self.calc_centerline_residence_time(volume, self.flow)
        return crt

    def _calc_radius_array(self, n=10000):
        r_heat = self.D_heat/2.
        r_arr = np.linspace(0, r_heat, n)
        return r_arr

    def _calc_u_velocity(self):
        r_arr = self._calc_radius_array()
        r_heat = self.D_heat/2.
        u = (2.0*self.flow*(1-(r_arr/r_heat)**2)) / (np.pi*r_heat**2)
        return u

    def _calc_t_parameter(self):
        u = self._calc_u_velocity()
        t = self.l_heat / u
        return t

    def _calc_E_parameter(self):
        r_heat = self.D_heat/2.
        t = self._calc_t_parameter()
        theta  = (np.pi * r_heat**2 * self.l_heat)/self.flow
        E = (0.5* theta**2)/(t**3)
        return E

    def _calc_Dt_parameter(self):
        t = self._calc_t_parameter()
        Dt = t[1:] - t[:-1]
        Dt = Dt[:-1]
        return Dt

    def get_avg_res_time_rings(self):
        t = self._calc_t_parameter()
        E = self._calc_E_parameter()
        Dt = self._calc_Dt_parameter()
        # Divide into rings
        avres1 = (sum(t[1:1000]*E[1:1000]*Dt[0:999])
                  / sum(E[1:1000]*Dt[0:999]))
        avres2 = (sum(t[1000:2000]*E[1000:2000]*Dt[1000:2000])
                  / sum(E[1000:2000]*Dt[1000:2000]))
        avres3 = (sum(t[2000:3000]*E[2000:3000]*Dt[2000:3000])
                  / sum(E[2000:3000]*Dt[2000:3000]))
        avres4 = (sum(t[3000:4000]*E[3000:4000]*Dt[3000:4000])
                  / sum(E[3000:4000]*Dt[3000:4000]))
        avres5 = (sum(t[4000:5000]*E[4000:5000]*Dt[4000:5000])
                  / sum(E[4000:5000]*Dt[4000:5000]))
        avres6 = (sum(t[5000:6000]*E[5000:6000]*Dt[5000:6000])
                  / sum(E[5000:6000]*Dt[5000:6000]))
        avres7 = (sum(t[6000:7000]*E[6000:7000]*Dt[6000:7000])
                  / sum(E[6000:7000]*Dt[6000:7000]))
        avres8 = (sum(t[7000:8000]*E[7000:8000]*Dt[7000:8000])
                  / sum(E[7000:8000]*Dt[7000:8000]))
        avres9 = (sum(t[8000:9000]*E[8000:9000]*Dt[8000:9000])
                  / sum(E[8000:9000]*Dt[8000:9000]))
        avres10 = (sum(t[9000:9998]*E[9000:9998]*Dt[9000:])
                   / sum(E[9000:9998]*Dt[9000:]))
        avg_res = [avres1, avres2, avres3, avres4, avres5,
                   avres6, avres7, avres8, avres9, avres10]
        return avg_res

    def get_rings(self):
        r_heat = self.D_heat/2.
        # More nonsense
        rakt = np.zeros(10)
        Area = np.zeros(10)
        for i in range(0, 10):
            rakt[i] = (i+1)*r_heat/10.
            if i == 0:
                Area[i] = np.pi*rakt[i]**2
            else:
                Area[i] = np.pi * (rakt[i]**2 - rakt[i-1]**2)
        Area0 = np.pi * r_heat**2
        return Area, Area0
