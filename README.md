# A first experiment with AI agents in numerical geometry

Working notes by **G. Bruno De Luca** on a first attempt to use research agents on a four-dimensional numerical geometry problem. The short agent-assisted investigation builds on earlier mathematics, the [original three-dimensional study](https://arxiv.org/abs/2501.00093), existing code and earlier numerical campaigns. The four-dimensional version of the training code was developed earlier by G. Bruno De Luca with Albert Law, using Codex.

The scientific record is frozen. Both the neural metric models and the agent orchestration were exploratory. The notebook records partial results and the difficulties they exposed.

## Read online or locally

The published notebook is at https://gbdl.github.io/4D-website/. To read a local copy, run from the repository root:

```sh
python3 -m http.server 8787 --bind 127.0.0.1
```

Open **http://127.0.0.1:8787/**. The local server lets the browser load the saved JSON files. No account, API key or model service is needed. The page combines coordinate-domain diagrams, matched maps and profiles, experiment comparisons, and geometric measurements.

## The two results

- **Local metric:** R4-022 joint-best, step 402034. Small tested Einstein residuals, with unresolved geometric closure.
- **Quotient metric:** R4-087 joint-best, step 26838. Matching is built into its tensor prescription and supported by numerical checks; Einstein residuals remain in the background field and transition regions.

These states have different training histories. Neither meets the combined accuracy and compatibility goal. The residual `E = ||Ric(g) + 3g||_g / 6` measures failure of Einstein's equation, not distance to an exact solution.

The main experiment explorer contains **79 completed training-and-audit workflows from 80 submitted**, one node failure, and 16 planned continuations that were not released. The later audit supplement describes additional checks and short polishing experiments on the frozen states, within the same scientific cutoff.

## Files

| Material | Location |
|---|---|
| Illustrated notebook | [index.html](index.html) |
| Data dictionary and sample scopes | [data/README.md](data/README.md) |
| Saved evidence and numerical tables | [data/](data/), [data/tables/](data/tables/) |
| Frozen parameters and CPU evaluator | [reproduce/README.md](reproduce/README.md) |
| Executable checks and measured outputs | [verification/README.md](verification/README.md) |

Development panels, later searches, face samples and ridge catalogues cover different points. Their maxima are observed values, not uniform bounds; their sample frequencies are not volume fractions. Some historical entries are summary evidence from larger arrays not distributed here. The data dictionary identifies those cases.

## Reproduce selected checks and figures

These commands need only Python's standard library:

```sh
python3 tools/export_evidence.py
python3 tools/build_audit_supplement.py
python3 tools/update_audit_display.py
python3 tools/check_geometry.py
```

The exporters check the bundled statistics, paired comparisons and audit supplement and regenerate their tables and core-area figure. The geometry script compares explicit copy membership, cusp inversion and analytic black-hole formulas. Neither evaluates a learned metric.

To regenerate the scientific figures:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
.venv/bin/python tools/regenerate_figures.py
```

The default output is `.local/regenerated-figures/`. Use `--output-dir assets/figures` only when deliberately replacing the saved figures. No additional scientific samples are created.

The [numerical companion](reproduce/README.md) has separate Python 3.12 installation instructions. Its `check.py` reconstructs diagnostics from saved derivatives and tests analytic controls; `check.py --metrics` also evaluates the frozen neural metrics at the bundled fixture. The measured CPU time, memory and tensor differences are in [evaluator.json](verification/evaluator.json). This small fixture is intended for reproduction, not to repeat the complete audit.

## Human and agent contributions

G. Bruno De Luca directed both efforts and evaluated their disagreements. **Codex (OpenAI)** was the builder: it translated formulas, implemented the two constructions, prepared and submitted the training experiments, analyzed results and drafted the account. **Claude Code (Anthropic)** independently reconstructed the metrics from their weights and audited them. The page records findings and corrections on both sides; its episodes are decision summaries rather than verbatim conversations.

The numerical tests provide the evidence. Agreement between agents is not an independent scientific test, and no controlled comparison of agent arrangements or research speedup was performed.

## Publication, citation and licensing

This repository is the source of the public site. GitHub Pages publishes directly from the root of `main`, as described in [DEPLOYMENT.md](DEPLOYMENT.md). Once enabled, pushes to `main` update the site automatically. The `.gitignore` excludes local environments, caches and internal assessment notes from normal Git staging.

For a correction, include the state, chart, coordinates, software versions and exact command. Use [CITATION.cff](CITATION.cff) for attribution and cite the original paper for the underlying study.

Code is **GPL-3.0-or-later** ([LICENSE](LICENSE)). Controlled prose, figures, data, weights and note content are **CC BY 4.0** ([LICENSE-CONTENT](LICENSE-CONTENT)). Referenced papers and software dependencies retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
