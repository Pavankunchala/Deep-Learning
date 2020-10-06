#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:45:32 2020

@author: pavankunchala
"""

import deepchem as dc
import numpy as np

tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()