#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter


from plotter_class import PlotFigure


data = pd.read_csv("/develop_ws/bag_files/00000085_synced_imu_rcou_data.csv")

data_fig = PlotFigure(nrows=2, sharex=True)
data_fig.add_data(0, data["t_s"], data["GyrX"], alpha=0.3)
data_fig.add_scatter(0, data["t_s"], data["GyrX"], alpha=0.7)
data_fig.add_data(1, data["t_s"], data["C1"], alpha=0.3)
data_fig.add_scatter(1, data["t_s"], data["C1"], alpha=0.7)

plt.show()