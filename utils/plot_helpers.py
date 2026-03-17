"""Shared matplotlib helpers for Streamlit pages."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE   = "#4a90e2"
ORANGE = "#f97316"
GREEN  = "#22c55e"
RED    = "#ef4444"
PURPLE = "#8b5cf6"


def new_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(True, alpha=0.25, color="#d1d5db")
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax


def style_ax(ax, title="", xlabel="", ylabel=""):
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1f2937")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#374151")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#374151")
    ax.tick_params(labelsize=9, colors="#4b5563")
