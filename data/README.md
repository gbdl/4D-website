# Saved evidence

Scientific cutoff: **23 October 2026**. The common comparison panels and matched profiles were assembled on 20 October. The later audit adds distinct searches, ridge measurements, cusp limits, core areas and integrals. It does not replace the earlier panel values.

## Files

- `metrics.json`: the **Local metric** (`raw022`, R4-022 joint-best, step 402034) and **Quotient metric** (`quotient087`, R4-087 joint-best, step 26838); matched maps/profiles, comparison-panel summaries, 96 face-pair measurements and 352 core-closure representatives per state.
- `experiments.json`: 79 completed experiments, terminal and selected states, saved learning curves and 63 paired comparisons. The cohort comprised 80 submitted runs, one node failure and 16 unreleased planned continuations.
- `audit.json`: 78 measurements from hash-recorded audit summaries, including finite searches, ordinary ridges and approximate volume-weighted estimates over a truncated domain.
- `audit-supplement.json`: eight additional source records through the same cutoff, with cusp-limit diagnostics, core-area measurements, global integrals and recorded corrections to earlier interpretations. Source hashes, extracted fields and sample scopes are retained.
- `tables/`: CSV exports and accessible HTML tables. Experiment terminal (`last`) and selected (`joint-best`) states are separate rows. Map/profile tables retain every saved grid location, including masked points.

## Units and definitions

Top-level metric/experiment statistics `rms`, `q99`, `maximum` and `bulk_rms` are **already percentages**. Map, profile and panel `E` values are dimensionless fractions: multiply by 100 for percentages. Scalar curvature has target −12. Core angles are in radians; face length errors are fractions; bending errors have inverse curvature-radius units. Audit-supplement records carry explicit units, including area, volume and integrated curvature.

`rms` is the largest RMS among the seven main regions and dedicated panels. `bulk_rms` uses only the seven main regions. `q99` is the largest regional/panel 99th percentile, not a pooled percentile. `maximum` is the largest sampled value. The historical score `Q=max(rms/0.3,q99,maximum)` uses percentage units and does not include a certification of geometric compatibility.

Maps are row-major: `axis0` is horizontal, `axis1` vertical. `mask=false` means outside the saved domain. Status 1 means valid; status 2 means a failed evaluation, which must remain visible. The supplied maps have no failed admissible points. `map_labels` gives coordinate names and fixed values. Radial profiles contain 513 points with a nested 257-point grid; the four native-cusp profiles contain 129 points each. The website does not interpolate additional scientific samples.

## What the samples mean

Most plotted points come from development panels; they are nonuniform and repeated. Their frequencies are not volume fractions, and observed maxima are not uniform bounds. The selected states were chosen from completed runs, not an untouched final qualification. Continuations share parents; comparing these two achieved states does not isolate one architecture change.

The ordinary-ridge audit and the filled-core catalogue are separate. The Local metric has an ordinary-ridge defect of about 0.326 degrees, larger than its filled-core defect. The Quotient metric retains nonzero floating-point compatibility values where measured; numerical-control scale is not rounded to exact zero.

The supplement's full-domain integrals use a physical-volume measure, randomized integration and cusp-tail treatment. They differ from worst-region panel RMS and from the earlier truncated estimates. Their statistical uncertainties are not rigorous total-error bounds. The Local metric's incompatible field is integrated over the stated coordinate domain, not assumed to define a unique smooth quotient metric. An Euler integral near the expected value cannot replace local geometry checks.

The integral summaries name the selected metrics but omit input-file hashes. Their state binding is indirect through the metric registry and separately hash-bound reconstruction records. This limitation is preserved in the supplement. The black-hole core area is a model reference, not the area of the unknown exact solution.

## Reproduce reductions and figures

From the repository root:

```sh
python3 tools/export_evidence.py
python3 tools/build_audit_supplement.py
python3 tools/regenerate_figures.py
```

The first two commands use the standard library and bundled evidence. They reproduce numerical reductions, verify paired settings and ratios, and regenerate tables and mechanical check outputs. The supplement script also regenerates `assets/figures/core-area.svg`. The plotting script needs NumPy and Matplotlib and writes its saved-sample figures under `.local/regenerated-figures/`; add `--output-dir assets/figures` to replace the site's static figures deliberately. None of these commands evaluates a learned metric.

For maintainers with original arrays, `tools/export_evidence.py --source-root RESEARCH_TREE` reads and verifies those sources, including saved metric/Riemann tensors, induced metrics, extrinsic curvature and core incidence. The supplement's optional `--source-root AUDIT_PACKAGE` verifies its original summaries and corrections. These read-only source routes require inputs beyond this public bundle.

## Provenance and license

Checkpoint hashes, evaluated steps, source hashes, pool identities and sample counts are preserved where available. Numbered source records identify hash-bound original artifacts; they do not promise that every original array is distributed here. Execution locations and operational metadata are omitted.

Controlled data and figures are CC BY 4.0, attributed to G. Bruno De Luca. Code is GPLv3-or-later. The data exports and checks were prepared under human direction with Codex (OpenAI) and Claude Code (Anthropic).
