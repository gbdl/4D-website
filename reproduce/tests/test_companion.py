# SPDX-License-Identifier: GPL-3.0-or-later
"""Small loader/domain regression checks; scientific controls live in check.py."""
import importlib
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metric_demo import load_metric,geometry as G


def test_reject_modified_weight_export(tmp_path,monkeypatch):
    api=importlib.import_module('metric_demo.api')
    (tmp_path/'weights').mkdir()
    for suffix in ('.npz','.json'):
        shutil.copyfile(api.ROOT/'weights'/('local'+suffix),tmp_path/'weights'/('local'+suffix))
    p=tmp_path/'weights/local.npz';p.write_bytes(p.read_bytes()+b'changed')
    monkeypatch.setattr(api,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='hash mismatch'):api.load_metric('local')


def test_raw_coordinates_are_not_folded():
    model=load_metric('local')
    x=np.array([1.5,1.2,.5,.5])
    assert np.array_equal(np.asarray(model._model._folded(x)),x)
    with pytest.raises(ValueError,match='outside'):model.validate(x,'ordinary')


def test_actual_core_projection():
    model=load_metric('local')
    model.validate([0,0,0,0],'core')
    with pytest.raises(ValueError,match='outside'):model.validate([0,0,100,100],'core')
    with pytest.raises(ValueError,match='tip'):model.validate([G.RJ,0,0,0],'ordinary')


def test_canonical_site_classes_and_development():
    assert G.NSITES==37
    assert [sum(G.SITE_CUSP==i) for i in range(4)]==[4,9,12,12]
    for row in G.site_consistency():
        assert row['same_class']
        assert row['frame_spread']==0
        assert row['phase_spread']==0
        assert row['orthogonal']==0
