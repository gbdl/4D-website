#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate scientific PNG figures from bundled JSON. No models or JAX.

Adapted from the internal note's plotting code by G. Bruno De Luca and research
agents. Numerical evidence is frozen; this script draws it without resampling.
Requires NumPy and Matplotlib. Default output is .local/regenerated-figures.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local/regenerated-figures"
ROLES = ("raw022", "quotient087")
NAMES = {"raw022": "Local metric", "quotient087": "Quotient metric"}
COLORS = {"raw022":"#236b8e", "quotient087":"#d97920"}
RJ = 2.80701761247
CENTRES = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])

def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.dpi": 180})
    return plt


def save(fig, name):
    names = {"spatial":"matched-sections", "profiles":"matched-profiles", "profiles-supplement":"cusp-profiles", "gluing":"face-and-core-compatibility"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / (names.get(name,name) + ".png"), bbox_inches="tight", dpi=180)


def spatial(data):
    plt = plotting()
    from matplotlib.colors import LogNorm
    names = ("body", "core-copy30", "cusp0")
    labels = (("Ordinary section\n$x_2=x_3=0.5$", "$x_1$", "$z$"),
              ("Core: $\\tau=(2.8311,0)$", "$U$", "$V$"),
              ("Cusp 0: $t=2$\nbase $X_3=0.25$", "base $X_1$", "base $X_2$"))
    values = [100*np.asarray(data["states"][r]["maps"][n]["E"], float) for r in ROLES for n in names]
    finite = np.concatenate([v[np.isfinite(v)] for v in values])
    lo, hi = max(1e-5, float(finite.min())), float(finite.max())
    norm = LogNorm(vmin=10**np.floor(np.log10(lo)), vmax=10**np.ceil(np.log10(hi)))
    fig, axes = plt.subplots(3, 2, figsize=(6.1, 8.1), layout="constrained")
    for row, (name, (title, xl, yl)) in enumerate(zip(names, labels)):
        for col, role in enumerate(ROLES):
            m = data["states"][role]["maps"][name]
            a, b = np.asarray(m["axis0"]), np.asarray(m["axis1"])
            e = 100*np.asarray(m["E"], float)
            status = np.asarray(m["status"])
            cmap = plt.get_cmap("magma").copy()
            cmap.set_bad("#e4e4e4")
            ax = axes[row, col]
            im = ax.pcolormesh(a, b, np.ma.masked_where(status != 1, e), cmap=cmap, norm=norm, shading="auto")
            if np.nanmin(e) < 1 < np.nanmax(e):
                ax.contour(a, b, e, levels=[1.], colors=["cyan"], linewidths=.8)
            if np.any(status == 2):
                yy, xx = np.where(status == 2)
                ax.scatter(a[xx], b[yy], c="#ff00ff", marker="x", s=18)
            ax.set(xlabel=xl, ylabel=yl, title=NAMES[role] if row == 0 else "")
            ax.text(.03, .97, title, transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
            if name != "body":
                ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=axes, fraction=.035, pad=.025)
    cb.set_label("Local Einstein error $100E$ (%)")
    save(fig, "spatial")
    plt.close(fig)


def profiles(data):
    if not all("profiles" in data["states"][r] for r in ROLES):
        return
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(6.1, 5.5), layout="constrained")
    for role in ROLES:
        pp = data["states"][role]["profiles"]
        radial = pp["radial-quotient-transition"]
        x = np.asarray(radial["ordinary"], float)
        z = x[:, 0]
        valid = np.asarray(radial["status"]) == 1
        error = np.where(valid, 100*np.asarray(radial["E"], float), np.nan)
        scalar = np.where(valid, 100*(np.asarray(radial["scalar"], float)+12)/12, np.nan)
        for row in range(2):
            axes[row, 0].plot(z, error, color=COLORS[role], label=NAMES[role])
            axes[row, 1].plot(z, scalar, color=COLORS[role], label=NAMES[role])
    for row in range(2):
        axes[row, 0].set(xlabel="$z$", ylabel="$100E$ (%)")
        axes[row, 1].set(xlabel="$z$", ylabel="$100(S+12)/12$ (%)")
        axes[row, 0].axhline(1, c=".5", lw=.8, ls="--")
        axes[row, 1].axhline(0, c=".5", lw=.8)
    for ax in axes.flat:
        ax.grid(alpha=.18)
    for ax in axes[1]:
        ax.set_xlim(.8, 1.6)
    axes[0, 0].set_title("Full radial profile")
    axes[0, 1].set_title("Signed scalar residual")
    axes[1, 0].set_title("Transition zoom")
    axes[1, 1].set_title("Transition zoom")
    axes[0, 0].legend(loc="best")
    save(fig, "profiles")
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(6.1, 7.8), layout="constrained")
    for role in ROLES:
        pp = data["states"][role]["profiles"]
        radial = pp["radial-cell30"]
        z = np.asarray(radial["ordinary"], float)[:, 0]
        valid = np.asarray(radial["status"]) == 1
        axes[0, 0].plot(z, np.where(valid, 100*np.asarray(radial["E"], float), np.nan), color=COLORS[role], label=NAMES[role])
        axes[0, 1].plot(z, np.where(valid, 100*(np.asarray(radial["scalar"], float)+12)/12, np.nan), color=COLORS[role])
        for index in range(4):
            p = pp[f"cusp{index}"]
            valid = np.asarray(p["status"]) == 1
            ax = axes.flat[index+2]
            ax.semilogx(np.asarray(p["parameter"], float), np.where(valid, 100*np.asarray(p["E"], float), np.nan),
                        color=COLORS[role])
    axes[0, 0].set(title="Neutral radial line", xlabel="$z$", ylabel="$100E$ (%)")
    axes[0, 1].set(title="Neutral radial line", xlabel="$z$", ylabel="$100(S+12)/12$ (%)")
    axes[0, 0].legend()
    for index in range(4):
        ax = axes.flat[index+2]
        ax.set(title=f"Cusp {index}", xlabel="Inverted height $t$", ylabel="$100E$ (%)")
        ax.axhline(1, c=".5", lw=.8, ls="--")
    for ax in axes.flat:
        ax.grid(alpha=.18)
    save(fig, "profiles-supplement")
    plt.close(fig)


def gluing(data):
    if not all("physical_gluing" in data["states"][r] and "core_links" in data["states"][r] for r in ROLES):
        return
    plt = plotting()
    fig = plt.figure(figsize=(6.1, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for role in ROLES:
        s = data["states"][role]
        g = s["physical_gluing"]
        p = g["by_pair"]
        axes[0].plot([x["pair"] for x in p], [100*x["length_max"] for x in p],
                     color=COLORS[role], label=NAMES[role], lw=1)
        axes[1].plot([x["pair"] for x in p], [x["bend_max"] for x in p],
                     color=COLORS[role], lw=1)
        angles = np.asarray(s["core_links"]["signed_angle_defect"])*180/np.pi
        axes[2].plot(np.arange(len(angles)), angles, color=COLORS[role], label=NAMES[role], lw=1)
    axes[0].set(xlabel="External face-pair index", ylabel="Max length mismatch (%)")
    axes[1].set(xlabel="External face-pair index", ylabel="Max bending mismatch")
    axes[2].set(ylabel="Angle deficit (degrees)", xlabel="Exact-core representative")
    axes[0].legend(loc="center right")
    for index in (0, 1):
        axes[index].set_yscale("symlog", linthresh=1e-12)
        axes[index].set_yticks([0, 1e-12, 1e-9, 1e-6, 1e-3] + ([1e-1] if index == 0 else []))
    groups = data["states"][ROLES[0]]["core_links"]["stratum"]
    centers, names = [], []
    for group in dict.fromkeys(groups):
        indices = np.flatnonzero(np.asarray(groups) == group)
        centers.append(float(np.mean(indices)))
        names.append(f"{len(indices)} {group}s")
        if indices[0]:
            axes[2].axvline(indices[0]-.5, c=".7", lw=.7)
    axes[2].set_xticks(centers, names)
    axes[2].axhline(0, c=".5", lw=.5)
    for ax in axes:
        ax.grid(alpha=.18)
    save(fig, "gluing")
    plt.close(fig)


def fold_long(x):
    return np.where(x < 0, -x, np.where(x <= 1, x,
                    np.where(x <= 2, 2 - x, x - 2)))


def height(x1, x2, x3):
    u = np.stack(np.broadcast_arrays(np.abs(x1), fold_long(x2),
                                    fold_long(x3)), axis=-1)
    return np.sqrt(np.maximum(0, np.max(
        1 - np.sum((u[..., None, :] - CENTRES) ** 2, axis=-1), axis=-1)))


def geometry():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(6.45, 2.8),
                             constrained_layout=True)
    ax = axes[0]
    x = np.linspace(-1, 1, 1001)
    h = height(x, 0*x, 0*x)
    ax.fill_between(x, h, RJ, color="#d6e6f5")
    ax.fill_between(x, 1, RJ, color="#83b6d8", alpha=.6)
    ax.plot(x, h, color="#163b55", lw=1.6)
    ax.axhline(1, color="#345b72", ls="--", lw=.8)
    ax.axhline(RJ, color="#a0432c", lw=1.3)
    ax.axvline(0, color="white", lw=1)
    ax.plot(0, 0, "o", color="#a0432c", ms=5)
    ax.text(0, 1.9, "filled cap", ha="center")
    ax.text(.05, .15, "finite ideal\nvertex", fontsize=8)
    ax.text(.96, RJ+.06, r"$z=r_j$", ha="right", color="#a0432c")
    ax.set(xlim=(-1, 1), ylim=(0, 3.05), xlabel=r"$x_1$", ylabel=r"$z$",
           title=r"(a) Slice $x_2=x_3=0$")
    ax = axes[1]
    x1, x2 = np.meshgrid(np.linspace(-1, 1, 401),
                         np.linspace(-1, 3, 801))
    mask = height(x1, x2, np.zeros_like(x1)) <= .55
    ax.pcolormesh(x1, x2, mask, shading="auto",
                  cmap=ListedColormap(["#f3f3f3", "#83b6d8"]), rasterized=True)
    ax.contour(x1, x2, mask.astype(float), levels=[.5],
               colors="#163b55", linewidths=.7)
    ax.axvline(0, color="#777777", lw=.7, ls=":")
    for v in (0, 1, 2):
        ax.axhline(v, color="#777777", lw=.7, ls=":")
    ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$",
           title=r"(b) $z=0.55,\ x_3=0$")
    ax.text(.03, .03, "blue: admitted by all\nhemisphere inequalities",
             transform=ax.transAxes, fontsize=8,
             bbox={"fc": "white", "ec": "none", "alpha": .9})
    ax = axes[2]
    angle = np.linspace(0, 2*np.pi, 361)
    for z, color in ((1, "#83b6d8"), (.95*RJ, "#345b72"),
                     (.99*RJ, "#a0432c")):
        rho = np.sqrt(4*RJ*(RJ-z)/3)
        ax.plot(rho*np.cos(angle), rho*np.sin(angle), color=color,
                lw=1.5, label=fr"$z/r_j={z/RJ:.2f}$")
    ax.plot(0, 0, "o", color="black", ms=4)
    ax.annotate("exact core", (0, 0), (.5, .7),
                arrowprops={"arrowstyle": "-", "color": "black"}, fontsize=8)
    ax.set_aspect("equal")
    ax.set(xlabel=r"$U=\rho\cos\theta$", ylabel=r"$V=\rho\sin\theta$",
           title="(c) Core chart")
    ax.legend(loc="lower left", fontsize=8, framealpha=.92)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT/"geometry-charts.png", dpi=180)
    plt.close(fig)



def experiments(data):
    plt = plotting()
    colors = {"Raw":"#236b8e", "Periodic":"#6d7d53", "Quotient":"#d97920"}
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    for ax,state,label in zip(axes,("last","joint-best"),("Terminal states","Selected joint-best states")):
        for family,color in colors.items():
            rows=[r["states"][state] for r in data["records"].values() if r["family"]==family]
            ax.scatter([s["rms"] for s in rows],[s["maximum"] for s in rows],s=24,color=color,alpha=.75,label=family)
        ax.axvline(.3,c=".5",ls="--",lw=.7);ax.axhline(1,c=".5",ls="--",lw=.7)
        ax.set(xscale="log",yscale="log",xlabel="Worst regional RMS (%)",ylabel="Largest sampled E (%)",title=label)
        from matplotlib.ticker import NullFormatter
        ax.set_xlim(.17,7);ax.set_ylim(.35,22)
        ax.set_xticks([.2,.3,.5,1,2,3,5],["0.2","0.3","0.5","1","2","3","5"])
        ax.set_yticks([.5,1,2,5,10,20],["0.5","1","2","5","10","20"])
        ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.16);ax.legend(frameon=False)
    fig.suptitle("79 completed experiments · common development panels")
    save(fig,"experiment-panels");plt.close(fig)
    labels=list(dict.fromkeys(x["label"] for x in data["comparisons"]))
    readable={"raw_eta=0.01":"Core repair 0.01","raw_eta=0.1":"Core repair 0.1","raw_eta=1.0":"Core repair 1", "periodic_ridge":"Periodic ridge pressure", "periodic_newpool":"Periodic new pool", "quotient_stratification":"Quotient stratification", "quotient_collar=0.25":"Transition emphasis 0.25", "quotient_collar=1.0":"Transition emphasis 1"}
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout="constrained",sharey=True)
    for ax,key,title in zip(axes,("Q_ratio","bulk_RMS_ratio"),("Full-panel score ratio", "Worst main-region RMS ratio")):
        for i,label in enumerate(labels):
            vals=[p[key] for p in data["comparisons"] if p["label"]==label]
            # Small deterministic vertical offsets separate paired outcomes.
            offsets=np.linspace(-.13,.13,len(vals))
            ax.scatter(vals,i+offsets,s=20,color="#236b8e",alpha=.65)
            ax.plot([np.median(vals)],[i],marker="|",color="#a64e20",ms=16,mew=2)
        ax.axvline(1,color=".3",ls="--",lw=.8);ax.set_xscale("log");ax.set_title(title)
        ax.set_xlabel("Treatment / control · below 1 is smaller")
        ax.set_yticks(range(len(labels)),[readable[l] for l in labels]);ax.grid(axis="x",alpha=.15)
    save(fig,"paired-comparisons");plt.close(fig)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path)
    args=p.parse_args()
    if args.output_dir:OUTPUT=args.output_dir
    metrics=json.loads((ROOT/"data/metrics.json").read_text())
    trials=json.loads((ROOT/"data/experiments.json").read_text())
    geometry();spatial(metrics);profiles(metrics);gluing(metrics);experiments(trials)
    print("Regenerated seven PNG figures from bundled evidence.")
