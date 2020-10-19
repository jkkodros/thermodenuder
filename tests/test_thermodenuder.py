# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:42:55 2020

Test routines for the thermodenuder class and methods

@author: J Kodros
"""

import pytest
import numpy as np
from src import thermodenuder


class TestCalcVolume(object):
    def test_with_ones(self):