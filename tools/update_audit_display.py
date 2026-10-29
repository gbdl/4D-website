#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the notebook's integral table from saved audit evidence; no evaluations."""
import argparse
from html import escape
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
START='<!-- AUDIT_INTEGRALS_START -->'
END='<!-- AUDIT_INTEGRALS_END -->'

def integral_rows(data):
    specs=[('volume','Volume (curvature-radius⁴)',1,6),
           ('euler_integral_minus_2','Bulk Euler integral − 2',1,7),
           ('integrated_W2','∫‖W‖² dV (full tensor norm)',1,5),
           ('volume_weighted_E_RMS','Volume-weighted Einstein RMS (%)',100,5)]
    rows=[]
    for key,label,mult,digits in specs:
        cells=[]
        for role in ('raw022','quotient087'):
            v=data['global_integrals'][role][key]
            text=f'{mult*v["value"]:.{digits}f} ± {mult*v["se"]:.{digits}f}'
            cells.append('<td>'+escape(text)+'</td>')
        rows.append('<tr><th scope="row">'+escape(label)+'</th>'+''.join(cells)+'</tr>')
    return '\n'+'\n'.join(rows)+'\n'

def check_or_update(root,check=False):
    data=json.loads((root/'data/audit-supplement.json').read_text())
    path=root/'index.html';page=path.read_text()
    assert page.count(START)==page.count(END)==1
    first,tail=page.split(START);old,last=tail.split(END)
    new=integral_rows(data)
    if check:
        assert old==new,'Displayed integral table differs from bundled evidence'
    else:path.write_text(first+START+new+END+last)
    return len(data['global_integrals'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--check',action='store_true');a=p.parse_args()
    check_or_update(a.root,a.check)
    print('Verified integral display against saved evidence.' if a.check else 'Updated integral display from saved evidence.')
