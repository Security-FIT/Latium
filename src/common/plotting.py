"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Optional

BASELINE_COLOR = '#000000'


def setup_mpl_style(plt, *, grid_alpha: float = 0.25) -> None:
    plt.rcParams.update(
        {
            'figure.dpi': 150,
            'savefig.dpi': 220,
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'axes.grid': True,
            'grid.alpha': grid_alpha,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )


def plot_baseline_overlay(ax, layers, values, label: Optional[str] = None, *, color: str = BASELINE_COLOR) -> None:
    import matplotlib.patheffects as pe

    markevery = max(1, len(layers) // 12)
    line = ax.plot(
        layers,
        values,
        color=color,
        linewidth=1.8,
        linestyle='--',
        marker='o',
        markersize=2.4,
        markerfacecolor='white',
        markeredgecolor=color,
        markeredgewidth=0.8,
        markevery=markevery,
        label=label,
        zorder=4,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=3.2, foreground='white'), pe.Normal()])
