#!/usr/bin/env python3
"""Figures for the manuscript. Vector output, embedded directly in the rendered PDF."""
from __future__ import annotations

import base64
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.6, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _embed(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return ("<img style='width:100%' src='data:image/svg+xml;base64,"
            + base64.b64encode(buf.getvalue()).decode() + "'/>")


from cohort_table import parse as parse_cohort, with_peak   # noqa: E402


def fig_kernels(build, users=("tim", "U013", "IK5"), key=None):
    """Estimated impulse responses for representative participants.

    `key` maps working identifiers to published labels. The per-participant files on disk are
    always named by the working identifier, so the mapping has to be applied to the LEGEND rather
    than to the lookup — otherwise an anonymised run silently prints the working name on the plot.
    """
    key = key or {}
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    plotted = 0
    for u in users:
        p = os.path.join(build, f"g4_{u}.md")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        vals = re.findall(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s", txt)
        if len(vals) < 5:
            continue
        lag = np.array([float(a) for a, _ in vals])
        amp = np.array([float(b) for _, b in vals])
        ax.plot(lag, amp, lw=1.0, label=f"participant {key.get(u, u)}")
        plotted += 1
    if not plotted:
        return ""
    ax.set_xlabel("Time since dose (min)")
    ax.set_ylabel("Relative activity")
    ax.set_xlim(0, 360)
    ax.legend(frameon=False)
    return _embed(fig)


def fig_configured_vs_observed(build):
    rows = with_peak(parse_cohort(build))     # kernels with no mode carry no peak to plot
    if not rows:
        return ""
    cfg = np.array([r["cfg"] for r in rows])
    obs = np.array([r["obs"] for r in rows])
    flag = np.array([r["flag"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lim = (0, 120)
    ax.plot(lim, lim, color="0.6", lw=0.6, ls="--", zorder=1)
    ax.scatter(cfg[~flag], obs[~flag], s=16, facecolor="none", edgecolor="k",
               lw=0.7, zorder=3, label="reconciled records")
    ax.scatter(cfg[flag], obs[flag], s=16, marker="^", facecolor="none",
               edgecolor="0.45", lw=0.7, zorder=3, label="residual > 0.15")
    ax.set_xlabel("Configured peak (min)")
    ax.set_ylabel("Observed peak (min)")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    return _embed(fig)


def fig_distributions(build):
    rows = with_peak(parse_cohort(build))
    if not rows:
        return ""
    cfg = np.array([r["cfg"] for r in rows])
    obs = np.array([r["obs"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    bins = np.arange(0, 125, 5)
    ax.hist(cfg, bins=bins, histtype="step", lw=1.0, color="k", label="configured")
    ax.hist(obs, bins=bins, histtype="stepfilled", lw=0, color="0.8", label="observed")
    ax.hist(obs, bins=bins, histtype="step", lw=0.8, color="0.35")
    ax.set_xlabel("Peak time (min)")
    ax.set_ylabel("Participants")
    ax.legend(frameon=False)
    return _embed(fig)


def all_figures(build, key=None):
    return {
        "kernels": fig_kernels(build, key=key),
        "scatter": fig_configured_vs_observed(build),
        "dists": fig_distributions(build),
    }
