# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluate one point; coordinates must be valid for the selected chart."""
import argparse,json
import numpy as np
from . import load_metric
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--metric',choices=['local','compatible'],default='local')
p.add_argument('--chart',choices=['ordinary','core','cusp'],required=True)
p.add_argument('--point',type=float,nargs=4,required=True)
p.add_argument('--site',type=int)
p.add_argument('--diagnostics',action='store_true',help='differentiate the complete metric through second order')
a=p.parse_args();m=load_metric(a.metric)
try:
 result=m.diagnostics(a.point,a.chart,a.site) if a.diagnostics else {'g':m.metric(a.point,a.chart,a.site).tolist(),'valid':True}
except (ValueError,FloatingPointError) as error:
 p.exit(2,json.dumps({'valid':False,'error':str(error)})+'\n')
print(json.dumps(result,indent=2,allow_nan=False))
