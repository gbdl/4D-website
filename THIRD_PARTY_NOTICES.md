# Attribution and third-party notices

## Scientific work and supplied material

The scientific question, earlier work, numerical data and selected metric states are attributed to **G. Bruno De Luca**. The researcher directed the work with assistance from Codex (OpenAI) and Claude Code (Anthropic). The agents contributed implementation, analysis, reporting and critical checks; numerical controls and saved evidence support the scientific statements.

The metric companion adapts the independent reconstruction and numerical checks associated with the two-construction note. Its provenance file records hashes of the source kernels and both original checkpoints. It includes numeric-only parameter exports and selected original-code reference tensors. The repository's code grant is GPL-3.0-or-later; its controlled content grant is CC BY 4.0. Adaptations and limitations are described in the companion README.

## Referenced literature

These papers and essays are cited or linked, not relicensed by this repository:

- G. Bruno De Luca, *Machine Learning Gravity Compactifications on Negatively Curved Manifolds*, [arXiv:2501.00093](https://arxiv.org/abs/2501.00093).
- Michael T. Anderson, *Dehn filling and Einstein metrics in higher dimensions*, [arXiv:math/0303260](https://arxiv.org/abs/math/0303260).
- Richard H. Bamler, *Construction of Einstein metrics by generalized Dehn filling*, [published article](https://ems.press/content/serial-article-files/31831).
- Richard S. Sutton, *The Bitter Lesson* (2019), [linked essay](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf).

The earlier technical-note PDF is not included in this publication. Links and citations do not imply endorsement of this notebook by the cited authors.

## Software dependencies

The static website uses browser HTML, CSS, JavaScript, SVG, Canvas and MathML. It does not bundle a JavaScript chart library, a web-font package, an AI model service or a remote analytics script.

The optional numerical and plotting tools depend on separately installed packages. Their own distributions include the applicable notices:

| Package | Upstream license | Role |
|---|---|---|
| JAX and jaxlib | Apache-2.0 | CPU automatic differentiation and compilation |
| NumPy | BSD-3-Clause | Numerical arrays and independent contractions |
| SciPy | BSD-3-Clause | Transitive JAX dependency |
| ml_dtypes | Apache-2.0 | Transitive JAX dependency |
| opt_einsum | MIT | Transitive JAX dependency |
| Matplotlib | Matplotlib license, based on the PSF license | Optional static figure regeneration |
| pytest | MIT | Optional development checks |

These packages are not vendored into the public source tree. Pinned versions used for the numerical companion are listed in `reproduce/requirements-lock.txt`. Installing them does not change their licenses.

Scientific figures use Matplotlib's DejaVu Sans font; the upstream font includes Bitstream Vera and DejaVu notices. No editable font binaries are distributed as website assets.

## License texts

`LICENSE` contains the GNU General Public License version 3, copyright Free Software Foundation, Inc. Repository code is offered under that version **or any later version**, as stated in the README and source SPDX identifiers. The license document itself permits verbatim copying but not modification.

`LICENSE-CONTENT` identifies the Creative Commons Attribution 4.0 International legal code and the scope of the author-controlled content grant. The full canonical legal code governs; the short explanation in that file does not replace it.
