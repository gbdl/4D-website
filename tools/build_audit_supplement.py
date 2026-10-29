#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify compact saved audit evidence and regenerate its table and SVG.

Default operation uses only bundled JSON and the Python standard library.
--source-root AUDIT_PACKAGE imports the dated audit's results/ and FINDINGS.jsonl
read-only. No learned metric evaluation is performed by either operation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('raw022', 'quotient087')
LABELS = {'raw022': 'Local metric', 'quotient087': 'Quotient metric'}
EXPECTED = {
    'cusp_analytic': ('ws9_a3_cusps.json', '77a978349089429d68b752085c0576e1ee4092ff2c2d020320bdecb1eb9b9757'),
    'cusp_finite': ('r5_cusp_spectra.json', '1ffca5d2b8604fdb7f518f8dc66356f48b0dd16e270c518a2092c4a1b7d52861'),
    'core_area': ('ws9_a4_systole.json', 'ea8e5a3be0f9c846d77ae92bd8c36403a6e0990ee061a3a1299f6c7134bec05f'),
    'core_surface': ('r5_core_minimal.json', '1454f151f7b188a73386668ddba85e39a747b8cbc0f1229fec74700891e20913'),
    'integrals_raw': ('ws8_gloc.json', 'de53f1f5cdfbf77edf6a59ad3b8abf0ba9f2d34e6d9582ecbffeab6abbce593d'),
    'integrals_quotient': ('ws8_gquot.json', 'abeb43b9f1376ecb7342cedd2af9edee5abbff732987ff6b3e8039adae78d350'),
    'integrals_deterministic': ('r6_det_integrals.json', 'b80892a8ad48620df817625fe9caf785008f27577d436e84fc34ec4e65e669c3'),
}
AR_HASH = '3af8e2079d86f81783373d39c796e54d834aa89e07f86670fe21ce4ebccfdf1b'
ATTRIBUTION_FILES = {
    'audit_plan': ('spec/PLAN.md', '407d0b1b10055ee2eb44b4eebc555bf9d0dcf3a8690f7cccfe89f6b8dc062df9', '2026-10-21'),
    'audit_gate': ('redteam/G2_gate.md', '0ca2375a35eb744842cda4411e4db90e437293ba776f08b685e0a7b030910d03', '2026-10-22'),
}
BUILDER_COMMIT = 'b5e4b3fb5852c0cbd4c07912e36371ae0395fe6b'
PARAMETER_HASHES = {
    'raw022': '7e40f5cc2204331a996aacfc0fcbbf2e5e76a9cda39f8da409e8016dbcae8176',
    'quotient087': '8869e9cc57810662bca33dcd36006498b84e18312dd464d83a949aa7095edfbe',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def select(obj, keys):
    return {k: obj[k] for k in keys}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def import_sources(source):
    extracts, sources = {}, {}
    for key, (name, expected) in EXPECTED.items():
        path = source / 'results' / name
        raw = path.read_bytes()
        assert sha(raw) == expected, f'Unexpected source hash: {name}'
        doc = json.loads(raw)
        assert doc['date'][:10] <= '2026-10-23', name
        inputs = doc.get('inputs', {})
        hashes = [h for p, h in inputs.items() if p.endswith(('raw022.npz', 'quotient087.npz'))]
        if key.startswith('integrals_') and key != 'integrals_deterministic':
            assert inputs == {}, 'Preserve disclosure of indirect identity binding.'
        else:
            assert set(hashes) == set(PARAMETER_HASHES.values()), name
        sources[key] = {
            'artifact': name, 'sha256': expected, 'bytes': len(raw),
            'date': doc['date'], 'source_version': doc.get('git_commit', ''),
            'check_id': doc['check_id'],
            'parameter_identity_binding': 'indirect: named metric registry and separately hash-bound reconstructions'
            if not hashes else 'explicit parameter-export hashes in the original source record',
            'parameter_export_sha256': PARAMETER_HASHES if hashes else None,
        }
        v = doc['values']
        if key == 'cusp_analytic':
            extracts[key] = {r: select(v[r], ['E_inf_closed', 'E_inf_median', 'E_inf_max', 'E_inf_min',
                'E_inf_argmax_site', 'site_spread_per_cusp', 'max_abs_E1e8_minus_closed',
                'max_abs_Qnum1e7_minus_cusp_limit']) for r in ROLES}
            extracts[key]['controls'] = select(v['controls'], ['C1_gH_maxE', 'C2_max_abs_E_minus_closed',
                'C4_closed_f64_vs_mp'])
            extracts[key]['mp_max_abs_diff'] = v['mp_max_abs_diff']
        elif key == 'cusp_finite':
            extracts[key] = {r: select(v[r], ['E_inf', 'max_X_spread_t2g', 'site_spread_frobenius']) for r in ROLES}
        elif key == 'core_area':
            extracts[key] = {'bh_closed': select(v['bh_closed'], ['area']),
                **{r: select(v[r], ['area', 'rel_area']) for r in ROLES}}
        elif key == 'core_surface':
            extracts[key] = {r: select(v['states'][r], ['A_coord', 'A_min', 'A_min_K2n24',
                'coord_minus_min_rel', 'rel_Amin_vs_BH']) for r in ROLES}
        elif key in ('integrals_raw', 'integrals_quotient'):
            extracts[key] = select(v, ['metric', 'replicas', 't_cut', 'allocation'])
            extracts[key]['derived'] = select(v['derived'], ['V', 'chi', 'chi_minus_2', 'W2',
                'E_rms', 'int_E2', 'int_eS', 'int_eS2', 'identity_lhs', 'identity_rhs', 'identity_residual'])
            extracts[key]['regional_integrals'] = {
                r: {k: v['groups'][r][k] for k in ['vol', 'E2']} for r in ['body', 'cap']}
        elif key == 'integrals_deterministic':
            extracts[key] = {r: {'runs': {
                n: select(record, ['V', 'chi_m2', 'W2', 'int_E2', 'identity_lhs', 'identity_rhs'])
                for n, record in v[r]['runs'].items()}} for r in ROLES}
    records = [json.loads(line) for line in (source / 'FINDINGS.jsonl').read_text().splitlines() if line.strip()]
    ar = next(record for record in records if record.get('id') == 'AR-001')
    assert sha(canonical(ar)) == AR_HASH, 'Unexpected AR-001 record.'
    sources['corrections'] = {'artifact': 'AR-001', 'sha256': AR_HASH,
        'hash_method': 'SHA256 of the original JSON record, sorted keys and compact separators',
        'date': ar['ts'], 'source_version': 'corrections recorded in commit 3373aaf',
        'parameter_identity_binding': 'editorial record; not a metric measurement'}
    extracts['corrections'] = select(ar, ['id', 'ts', 'statement', 'notes'])
    excerpts = {}
    for key, (name, expected, date) in ATTRIBUTION_FILES.items():
        raw = (source/name).read_bytes()
        assert sha(raw) == expected, f'Unexpected attribution source: {name}'
        lines = raw.decode().splitlines()
        excerpts[key] = [line for line in lines if
            ('**Opus 5.5 at' in line or '**Sonnet 5.5 at' in line or '**Fable 5.1,' in line)] \
            if key == 'audit_plan' else [line for line in lines if line.startswith(('- Opus:', '- Sonnet:', '- Fable:'))]
        assert len(excerpts[key]) == 3
        sources[key] = {'artifact': name, 'sha256': expected, 'bytes': len(raw), 'date': date,
            'date_scope': 'Date of the selected model-role passage; the containing record includes later corrections through 23 October.',
            'parameter_identity_binding': 'Attribution only; not a metric measurement'}
    metadata = subprocess.check_output(['git', '-C', str(source), 'show', '-s',
        '--format=%H%n%aI%n%an%n%s', BUILDER_COMMIT]).decode().splitlines()
    assert metadata[:3] == [BUILDER_COMMIT, '2026-10-23T17:50:07+02:00', 'Codex']
    extracts['attribution'] = {'excerpts': excerpts, 'builder_commit_metadata': {
        'commit': metadata[0], 'date': metadata[1], 'author': metadata[2], 'subject': metadata[3]}}
    sources['builder_commit'] = {'artifact': BUILDER_COMMIT,
        'sha256': sha(canonical(extracts['attribution']['builder_commit_metadata'])),
        'hash_method': 'SHA256 of sorted compact JSON containing commit, date, author and subject.',
        'date': metadata[1], 'parameter_identity_binding': 'Attribution only; not a metric measurement'}
    audit = read(ROOT / 'data/audit.json')
    identities = audit['identities']
    for role in ROLES:
        assert identities[role]['audit_export_sha256'] == PARAMETER_HASHES[role]
    return {'schema': 1, 'scientific_snapshot': '2026-10-23', 'license': 'CC-BY-4.0',
        'purpose': 'Additional saved checks of the same two frozen states; no new metric evaluations.',
        'identities': identities, 'sources': sources, 'saved_extracts': extracts,
        **assemble(extracts)}


def estimate(record, unit):
    return {'value': record['mean'], 'se': record['se'], 'ci95_half_width': record['ci95'], 'unit': unit}


def assemble(x):
    cusp, area, integrals = {}, {}, {}
    reference = x['core_area']['bh_closed']['area']
    for role in ROLES:
        a, f = x['cusp_analytic'][role], x['cusp_finite'][role]
        cusp[role] = {'source': 'cusp_analytic', 'source_keys': ['values', role],
            'analytic_limit_E': {'maximum': a['E_inf_max'], 'median': a['E_inf_median'],
                'minimum': a['E_inf_min'], 'site_count': len(a['E_inf_closed']),
                'argmax_site': a['E_inf_argmax_site'], 'per_site': a['E_inf_closed'], 'unit': 'dimensionless E'},
            'finite_reconstruction': {'source': 'cusp_finite', 't': 1_000_000,
                'maximum_E_limit_estimate': f['E_inf']['max'],
                'unit': 'dimensionless E', 'scope': 'Limit estimated from t squared g at finite t; not a sampled Einstein-error maximum.'},
            'site_shape_spread_per_cusp': a['site_spread_per_cusp'],
            'shape_spread_definition': 'Frobenius differences of normalized cusp-limit shape matrices across images of the same cusp.',
            'scope': '37 cusp sites. Analytic E_infinity = abs(1 - inverse(Q)[0,0]) for the constant limiting tensor Q. These are not additional samples on the finite-depth profile plots.'}
        s = x['core_surface'][role]
        area[role] = {'source': 'core_area', 'source_keys': ['values', role, 'area'],
            'coordinate_area': x['core_area'][role]['area'], 'unit': 'curvature-radius squared',
            'relative_to_BH': x['core_area'][role]['area'] / reference - 1,
            'surface_check': {'source': 'core_surface', 'coordinate_area': s['A_coord'],
                'optimized_area': s['A_min'], 'relative_reduction': s['coord_minus_min_rel'],
                'alternative_resolution_area': s['A_min_K2n24'],
                'scope': 'Bounded local surface search, not a proof of a global minimum. Raw-field incompatibility remains.'}}
        key = 'integrals_raw' if role == 'raw022' else 'integrals_quotient'
        v = x[key]; d = v['derived']
        integrals[role] = {'source': key, 'source_keys': ['values', 'derived'],
            'volume': estimate(d['V'], 'curvature-radius to the fourth power'),
            'euler_integral_minus_2': estimate(d['chi_minus_2'], 'dimensionless'),
            'integrated_W2': estimate(d['W2'], 'dimensionless; full tensor contraction'),
            'volume_weighted_E_RMS': estimate(d['E_rms'], 'dimensionless E'),
            'einstein_volume_identity_difference': estimate(d['identity_lhs'], 'curvature-radius to the fourth power'),
            'replicas': v['replicas'], 'evaluated_points': v['replicas'] * sum(v['allocation'].values()),
            'tail_start_t': v['t_cut'],
            'measure': 'Physical volume measure, stratified scrambled-Sobol integration with analytic reference control variates and cusp-tail treatment.',
            'uncertainty': 'SE is the spread-based standard error across 16 randomized replicas; the 95% half-width uses 15 degrees of freedom. Neither is a rigorous total-error bound.',
            'domain': 'Stated half-open slab fundamental domain, body/cap/native-cusp decomposition and analytic tails beyond t=100. For the raw field, incompatibility prevents treating this as a unique quotient-domain integral.',
            'independent_crosscheck': {
                'source': 'integrals_deterministic',
                'status': 'reproduced for the raw state' if role == 'raw022' else 'inconclusive for quotient Euler/Weyl integrals',
                'note': 'Separate deterministic decompositions and resolution studies reproduce the raw-state totals.'
                    if role == 'raw022' else 'Affordable deterministic tensor rules under-resolve reflection collars. This is not a third successful confirmation; it does not refute the randomized estimates.'}}
    return {'cusp_limits': cusp,
        'core_area': {'reference': {'label': 'Black-hole reference', 'area': reference,
            'unit': 'curvature-radius squared', 'source': 'core_area',
            'scope': 'An analytic model comparison, not the unknown exact filling or an imposed target.'},
            'states': area, 'interpretation': 'The two learned geometries differ substantially in core area. These measurements alone do not determine which is closer to an exact Einstein filling.'},
        'global_integrals': integrals,
        'corrections': [
            {'topic': 'Where the quotient error occurs', 'side': 'research note',
                'earlier': 'The description emphasized transition regions.',
                'revised': 'The note now describes broad background error, with additional transition peaks.',
                'source': 'corrections', 'scope': 'Paraphrase of the recorded response and revision; not a transcript.'},
            {'topic': 'Geometric closure', 'side': 'research note',
                'earlier': 'Core closure measurements carried too much of the interpretation.',
                'revised': 'Ordinary-ridge defects are reported separately and are larger than the filled-core defects.',
                'source': 'corrections', 'scope': 'Paraphrase of the recorded response and revision; not a transcript.'},
            {'topic': 'A trial smoothing', 'side': 'audit',
                'earlier': 'A flat profile with a quoted 3.3% error was counted as a smoothing.',
                'revised': 'It was not C1 at its boundary; the figure was withdrawn as evidence for a completed repair.',
                'source': 'corrections', 'scope': 'Correction of an audit interpretation.'},
            {'topic': 'A stalled optimization', 'side': 'audit',
                'earlier': 'A polishing plateau was interpreted as an architectural limitation.',
                'revised': 'It is a limitation of the tested procedure at one checkpoint, not a demonstrated error floor of the model class.',
                'source': 'corrections', 'scope': 'Correction of an audit interpretation.'},
            {'topic': 'Linear versus nonlinear solver progress', 'side': 'authors response',
                'earlier': 'The response described the linear solve as still improving.',
                'revised': 'The recorded linear solve had converged to its stopping criterion; nonlinear runs were stopped early.',
                'source': 'corrections', 'scope': 'Correction to the response itself; this is not a claim that the resulting metric met the scientific target.'},
        ],
        'attribution': {'summary': 'Research decisions and direction: G. Bruno De Luca. Agents helped produce the research note and a separate audit; review of the audit changed interpretations on both sides.',
            'systems': [
                {'system': 'Codex', 'provider': 'OpenAI', 'role': 'Builder: implementation, experiments, analysis and research-note preparation.',
                    'source': 'builder_commit', 'recorded_model': 'Version not established by the retained records.',
                    'evidence_scope': 'The dated note-update commit records Codex as author; it does not establish every individual tool call or model version.'},
                {'system': 'Claude Code', 'provider': 'Anthropic', 'role': 'Separate audit, with delegated calculations and critical reviews.',
                    'sources': ['audit_plan', 'audit_gate'],
                    'recorded_model': 'The revised plan names Opus 5.5, Sonnet 5.5 and Fable 5.1 for different roles.',
                    'evidence_scope': 'Recorded assignments and a coordinator execution summary, not independently verified per-call model telemetry.'},
            ],
            'recorded_model_roles': [
                {'recorded_model': 'Opus 5.5', 'role': 'Mathematical and verification work in the revised audit plan.',
                    'plan_source': 'audit_plan', 'execution_record': 'The G2 summary records the Opus family for H1, H2, H5 and H9; it does not record a version per call.'},
                {'recorded_model': 'Sonnet 5.5', 'role': 'Engineering and finishing work in the revised audit plan.',
                    'plan_source': 'audit_plan', 'execution_record': 'The G2 summary records the Sonnet family for H3, H4, H6, H7 and H8; it does not record a version per call.'},
                {'recorded_model': 'Fable 5.1', 'role': 'Confined double-checks in the revised audit plan.',
                    'plan_source': 'audit_plan', 'execution_record': 'The G2 summary records the Fable family for its bounded review; it does not record a version per call.'},
            ],
            'scope': 'Recorded designations describe this first attempt; they are not a benchmark of the named systems, a full per-call inventory or evidence of a validated agent architecture.'},
        'caveats': [
            'Observed data and numerical controls do not certify a smooth global Einstein filling.',
            'Cusp-limit errors are distinct from maxima on saved finite-depth profiles.',
            'The model-reference core area is not the true-answer area; no universal residual-to-distance estimate is inferred.',
            'Euler-integral disagreement corroborates incompatibility only with correct topology, boundary/tail treatment, domain coverage, normalization and integration. Agreement cannot certify smoothness or Einstein accuracy.',
            'Volume-weighted RMS from full-domain integration is distinct from worst-region comparison-panel RMS and the earlier truncated C3 volume estimates.',
            'The integral sources omit input-file hashes. Their state binding is indirect through the named metric registry and separately hash-bound reconstruction records.',
            'Audit stages named R5/R6 here date 22–23 October and are not later training campaigns. No post-snapshot scientific measurements are included.',
        ]}


def close(a, b, atol=2e-12, rtol=3e-10):
    assert math.isclose(a, b, abs_tol=atol, rel_tol=rtol), (a, b)


def verify(data):
    assert data['scientific_snapshot'] == '2026-10-23'
    for key, (_, expected) in EXPECTED.items():
        assert data['sources'][key]['sha256'] == expected
        assert data['sources'][key]['date'][:10] <= '2026-10-23'
    assert data['sources']['corrections']['sha256'] == AR_HASH
    for key, (_, expected, _) in ATTRIBUTION_FILES.items():
        assert data['sources'][key]['sha256'] == expected
    meta = data['saved_extracts']['attribution']['builder_commit_metadata']
    assert meta['commit'] == BUILDER_COMMIT and meta['author'] == 'Codex'
    assert sha(canonical(meta)) == data['sources']['builder_commit']['sha256']
    original = read(ROOT / 'data/audit.json')
    for role in ROLES:
        assert data['identities'][role] == original['identities'][role]
        assert data['identities'][role]['audit_export_sha256'] == PARAMETER_HASHES[role]
    expected = assemble(data['saved_extracts'])
    for key, value in expected.items():
        assert data[key] == value, f'Derived supplement changed: {key}'
    checks = []
    for role in ROLES:
        c = data['cusp_limits'][role]['analytic_limit_E']; values = sorted(c['per_site'])
        assert len(values) == 37
        close(c['maximum'], max(values)); close(c['minimum'], min(values)); close(c['median'], values[18])
        assert c['argmax_site'] == c['per_site'].index(max(values))
        f = data['cusp_limits'][role]['finite_reconstruction']['maximum_E_limit_estimate']
        assert abs(f - c['maximum']) < 1e-6
        x = data['saved_extracts']; v = x['integrals_raw' if role == 'raw022' else 'integrals_quotient']['derived']
        close(v['chi']['mean'] - 2, v['chi_minus_2']['mean'])
        close(v['identity_lhs']['mean'], v['V']['mean'] - (8*math.pi**2/3 - v['W2']['mean']/24))
        close(v['identity_lhs']['mean'], v['identity_rhs']['mean'], atol=1e-11)
        # Mean replica RMS differs slightly from RMS of mean integrals; retain the reported estimator.
        pooled_rms = math.sqrt(v['int_E2']['mean']/v['V']['mean'])
        assert abs(pooled_rms-v['E_rms']['mean']) < 1e-7
        a = data['core_area']['states'][role]
        close(a['relative_to_BH'], x['core_area'][role]['rel_area'])
        s = a['surface_check']; close(s['relative_reduction'], (s['coordinate_area']-s['optimized_area'])/s['coordinate_area'])
        checks.append({'role': role, 'checkpoint_identity': 'pass', 'cusp_site_reduction': 'pass',
            'core_area_reduction': 'pass', 'integral_normalization': 'pass',
            'RMS_of_mean_integrals_minus_reported_replica_mean': pooled_rms-v['E_rms']['mean']})
    return checks


def tables(data):
    rows = []
    for role in ROLES:
        identity = data['identities'][role]
        for name, value, unit, uncertainty, source, scope in [
            ('maximum cusp-limit E', 100*data['cusp_limits'][role]['analytic_limit_E']['maximum'], 'percent', '', 'cusp_analytic', 'Analytic limit over 37 sites; not a finite-profile maximum.'),
            ('coordinate core area', data['core_area']['states'][role]['coordinate_area'], 'curvature-radius squared', '', 'core_area', 'Area of the coordinate core; raw compatibility remains unresolved.'),
            ('core area relative to BH', 100*data['core_area']['states'][role]['relative_to_BH'], 'percent', '', 'core_area', 'Reference comparison, not error against a true solution.'),
            *[(name, value['value']*(100 if key == 'volume_weighted_E_RMS' else 1),
               'percent' if key == 'volume_weighted_E_RMS' else value['unit'],
               value['se']*(100 if key == 'volume_weighted_E_RMS' else 1),
               data['global_integrals'][role]['source'], 'Randomized integration with cusp tails; uncertainty is one statistical SE.')
              for key, name in [('volume', 'volume'), ('euler_integral_minus_2', 'Euler integral minus 2'),
                ('integrated_W2', 'integrated full-tensor Weyl norm squared'), ('volume_weighted_E_RMS', 'volume-weighted Einstein RMS')]
              for value in [data['global_integrals'][role][key]]],
        ]:
            rows.append({'construction': LABELS[role], 'state': identity['id'], 'step': identity['step'],
                'measurement': name, 'value': value, 'unit': unit, 'standard_error': uncertainty,
                'scope': scope, 'source': source, 'source_sha256': data['sources'][source]['sha256'],
                'checkpoint_sha256': identity['checkpoint_sha256']})
    path = ROOT/'data/tables/audit-supplement.csv'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    headings = ''.join(f'<th scope="col">{html.escape(k.replace("_", " "))}</th>' for k in rows[0])
    body = ''.join('<tr>' + ''.join('<td>'+html.escape(str(v))+'</td>' for v in row.values()) + '</tr>' for row in rows)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Additional saved audit evidence</title><style>body{font:16px/1.55 system-ui;color:#253d42;background:#faf8f2;margin:2rem}a{color:#286d78}table{border-collapse:collapse;font-size:.85rem}td,th{padding:.55rem;border-bottom:1px solid #d8d6cd;text-align:left}th{background:#eeeae0}.scroll{overflow:auto}main{max-width:90rem}li{margin:.4rem 0}</style>
<main><a href="../../index.html">Return to the notebook</a><h1>Additional checks of the same two states</h1>
<p>Scientific snapshot: 23 October 2026. These saved audit measurements are different from the common comparison panels. No new model evaluation was needed to export this table.</p>
<p><a href="audit-supplement.csv" download>Download CSV</a> · <a href="../audit-supplement.json" download>Download evidence and scopes (JSON)</a></p>
<p>The black-hole reference core area is ''' + str(data['core_area']['reference']['area']) + ''' in curvature-radius squared units. It is an analytic comparison, not the unknown exact filling.</p>
<div class="scroll" tabindex="0"><table><caption>Fourteen saved measurements; uncertainties shown are one statistical standard error where available.</caption><thead><tr>''' + headings + '</tr></thead><tbody>' + body + '''</tbody></table></div>
<h2>Limits of these checks</h2><ul>''' + ''.join('<li>'+html.escape(v)+'</li>' for v in data['caveats']) + '''</ul>
<h2>Revisions after checking</h2><p>Retrospective summaries of dated records, not quotations or transcripts.</p><ul>''' + ''.join('<li><strong>'+html.escape(c['topic'])+' ('+html.escape(c['side'])+').</strong> '+html.escape(c['revised'])+'</li>' for c in data['corrections']) + '</ul></main></html>\n'
    (ROOT/'data/tables/audit-supplement.html').write_text(page)
    return len(rows)


def figure(data):
    reference = data['core_area']['reference']['area']
    def x(v): return 180 + (v-.30)/.09*390
    ticks = ''.join(f'<line x1="{x(v):.2f}" x2="{x(v):.2f}" y1="188" y2="194" stroke="#727e7e"/><text x="{x(v):.2f}" y="213" text-anchor="middle">{v:.2f}</text>' for v in [.30,.32,.34,.36,.38,.39])
    rows = []
    for role, y, color in [('raw022', 107, '#286d78'), ('quotient087', 157, '#b55e34')]:
        a = data['core_area']['states'][role]; p = x(a['coordinate_area'])
        rows.append(f'<text x="20" y="{y+5}" font-size="15" fill="{color}">{LABELS[role]}</text><line x1="{x(reference):.2f}" x2="{p:.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/><circle cx="{p:.2f}" cy="{y}" r="6" fill="{color}"/><text x="{p:.2f}" y="{y-14}" text-anchor="middle" fill="{color}" font-size="13">{a["coordinate_area"]:.5f} ({100*a["relative_to_BH"]:+.2f}%)</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 280" role="img" aria-labelledby="title desc"><title id="title">The two core areas differ</title><desc id="desc">The local metric coordinate core area is 0.37265, 5.42 percent above the black-hole reference 0.35349. The quotient metric area is 0.32668, 7.58 percent below. The analytic reference is not the unknown exact solution. The plotted axis begins at 0.30.</desc><rect width="640" height="280" rx="8" fill="#faf8f2"/><g font-family="system-ui, sans-serif" font-size="12" fill="#34474a"><text x="20" y="29" font-size="18" font-weight="600">The two core areas differ</text><text x="20" y="51">Saved coordinate-core areas · same two frozen states</text><line x1="{x(reference):.2f}" x2="{x(reference):.2f}" y1="75" y2="188" stroke="#778481" stroke-dasharray="4 4"/><text x="{x(reference):.2f}" y="71" text-anchor="middle" font-size="11">BH reference: {reference:.5f}</text>{''.join(rows)}<line x1="180" x2="570" y1="188" y2="188" stroke="#727e7e"/>{ticks}<text x="375" y="234" text-anchor="middle">Area / curvature-radius² (axis starts at 0.30)</text><text x="20" y="261" font-size="12">The reference is an analytic model, not the unknown exact filling.</text></g></svg>\n'''
    (ROOT/'assets/figures/core-area.svg').write_text(svg)
    def xm(v): return 40 + (v-.30)/.09*280
    mobile_rows = []
    for role, y, color in [('raw022', 148, '#286d78'), ('quotient087', 214, '#b55e34')]:
        a = data['core_area']['states'][role]; p = xm(a['coordinate_area'])
        mobile_rows.append(f'<text x="20" y="{y-31}" font-size="15" fill="{color}">{LABELS[role]}</text><text x="20" y="{y-12}" fill="{color}">{a["coordinate_area"]:.5f} ({100*a["relative_to_BH"]:+.2f}%)</text><line x1="{xm(reference):.2f}" x2="{p:.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/><circle cx="{p:.2f}" cy="{y}" r="6" fill="{color}"/>')
    mobile_ticks = ''.join(f'<line x1="{xm(v):.2f}" x2="{xm(v):.2f}" y1="250" y2="256" stroke="#727e7e"/><text x="{xm(v):.2f}" y="275" text-anchor="middle">{v:.2f}</text>' for v in [.30,.33,.36,.39])
    mobile = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 355" role="img" aria-labelledby="title desc"><title id="title">The two core areas differ</title><desc id="desc">Local metric: 0.37265, 5.42 percent above the black-hole model reference 0.35349. Quotient metric: 0.32668, 7.58 percent below. The model reference is not the unknown exact filling. Axis starts at 0.30.</desc><rect width="360" height="355" rx="8" fill="#faf8f2"/><g font-family="system-ui, sans-serif" font-size="13" fill="#34474a"><text x="20" y="28" font-size="18" font-weight="600">The two core areas differ</text><text x="20" y="50">Saved coordinate-core areas</text><text x="20" y="69">Same two frozen states</text><text x="20" y="91">BH reference: {reference:.5f}</text><line x1="{xm(reference):.2f}" x2="{xm(reference):.2f}" y1="105" y2="250" stroke="#778481" stroke-dasharray="4 4"/>{''.join(mobile_rows)}<line x1="40" x2="320" y1="250" y2="250" stroke="#727e7e"/>{mobile_ticks}<text x="180" y="297" text-anchor="middle">Area / curvature-radius² · axis from 0.30</text><text x="20" y="324">The reference is an analytic model,</text><text x="20" y="343">not the unknown exact filling.</text></g></svg>\n'''
    (ROOT/'assets/figures/core-area-mobile.svg').write_text(mobile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Read-only audit package containing results/ and FINDINGS.jsonl')
    args = parser.parse_args()
    path = ROOT/'data/audit-supplement.json'
    if args.source_root:
        write(path, import_sources(args.source_root))
    data = read(path)
    checks = verify(data)
    count = tables(data); figure(data)
    receipt = {'schema': 1, 'scientific_snapshot': '2026-10-23', 'status': 'passed', 'checks': checks,
        'table_rows': count, 'source_records': len(data['sources']),
        'source_hash_validation': 'Original files verified during import.' if args.source_root else 'Bundled source identities checked against frozen expected hashes; original files not read.',
        'scope': 'Saved scalar/array reductions and provenance checks only; no learned metric evaluation or new integration.',
        'artifacts_sha256': {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in [path,
            ROOT/'data/tables/audit-supplement.csv', ROOT/'data/tables/audit-supplement.html', ROOT/'assets/figures/core-area.svg',
            ROOT/'assets/figures/core-area-mobile.svg']}}
    write(ROOT/'verification/audit-supplement.json', receipt)
    print(json.dumps({'status': 'passed', 'table_rows': count, 'source_records': len(data['sources']),
        'scope': receipt['scope']}, indent=2))


if __name__ == '__main__':
    main()
