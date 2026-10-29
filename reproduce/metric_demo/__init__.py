# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen CPU/float64 companion for two selected numerical metric states."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
from .api import load_metric, Metric, geometry
__all__ = ['load_metric', 'Metric', 'geometry']
