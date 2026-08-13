#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rga_plots.py — figures for RGA/TEY measurements.

PlotMixin is mixed into rga_batch.RGAMeasurement, so every method here reaches
the data through self (self.time, self.pressure, self.tey_signal, ...). Nothing
in this module imports rga_batch.

The Agg backend is selected on import because the consumer runs headless.
"""

import matplotlib
matplotlib.use("Agg")   # headless; must precede any pyplot import

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import (ScalarFormatter, LogLocator, NullFormatter,
                               LogFormatterSciNotation)
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Tick helpers
# ---------------------------------------------------------------------------

def _nice_step(span, target_n):
    raw = span / target_n
    if raw <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(raw))
    opts = np.array([1, 2, 2.5, 5, 10])
    return float(opts[np.argmin(np.abs(opts * mag - raw))] * mag)


def _nice_ticks(vmin, vmax, target_n=6):
    step = _nice_step(vmax - vmin, target_n)
    t = np.arange(np.ceil(vmin / step) * step, vmax + step * 0.1, step)
    return np.unique(np.round(t).astype(int))


def _nice_dose_ticks(dmax, target_n=5):
    step = _nice_step(dmax, target_n)
    step = int(round(step)) if step >= 1 else round(step, 2)
    t = np.arange(0, dmax + step * 0.1, step)
    return t[t <= dmax * 1.05]


# ---------------------------------------------------------------------------
# Plot methods
# ---------------------------------------------------------------------------

class PlotMixin:
    """Plotting half of RGAMeasurement."""

    def _shade(self, ax):
        if hasattr(self, "open_time"):
            ax.axvspan(self.open_time, self.close_time, color="gold",
                       alpha=0.20, lw=0, label="beam on")
            for lo, hi in (self._bg_off1, self._bg_off2):
                ax.axvspan(lo, hi, color="grey", alpha=0.20, lw=0)

    def plot_tey(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3))
        ax.plot(self.tey_time, self.tey_signal * 1e9, lw=1, color="tab:blue")
        self._shade(ax)
        ax.set_xlabel("time (s)"); ax.set_ylabel("TEY (nA)")
        ax.set_title(f"{self.sample_name} — TEY")
        return ax

    def plot_total(self, ax=None, raw=False):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3))
        ax.plot(self.time, np.nansum(self.pressure, axis=1), lw=1,
                color="tab:red",
                label="bg-subtracted" if self.is_corrected else "total")
        if raw and self.is_corrected:
            ax.plot(self.time, np.nansum(self._raw_pressure, axis=1), lw=1,
                    color="grey", alpha=0.7, label="raw")
            ax.legend(fontsize=8)
        self._shade(ax)
        ax.set_xlabel("time (s)"); ax.set_ylabel("Σ pressure (Torr)")
        ax.set_title(f"{self.sample_name} — total pressure")
        return ax

    def plot_spectrum(self, ax=None, mz_max=None, log=False):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3))
        spec = self.get_spectrum()
        ax.bar(self.mz, spec, width=0.8, color="tab:purple")
        if mz_max:
            ax.set_xlim(0, mz_max)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("m/z"); ax.set_ylabel("pressure (Torr)")
        ax.set_title(f"{self.sample_name} — spectrum (beam-on avg)")
        return ax

    def plot_imshow(self, ax=None, mz_max=None, vmax_pct=99.5, cmap=None):
        """Diverging map once corrected, since the negatives are real."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.5))
        if cmap is None:
            cmap = "RdBu_r" if self.is_corrected else "viridis"
        p = self.pressure
        sel = np.ones(self.n_mz, bool) if mz_max is None else self.mz <= mz_max
        v = np.nanpercentile(np.abs(p[:, sel]), vmax_pct)
        im = ax.imshow(p[:, sel].T, aspect="auto", origin="lower", cmap=cmap,
                       vmin=-v if self.is_corrected else 0, vmax=v,
                       extent=[self.time[0], self.time[-1],
                               self.mz[sel][0], self.mz[sel][-1]])
        ax.figure.colorbar(im, ax=ax, label="pressure (Torr)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("m/z")
        ax.set_title(f"{self.sample_name} — MS(t)")
        return ax

    def plot_summary(self, mz_max=100):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        self.plot_tey(axes[0, 0])
        self.plot_total(axes[0, 1], raw=True)
        self.plot_imshow(axes[1, 0], mz_max=mz_max)
        self.plot_spectrum(axes[1, 1], mz_max=mz_max)
        fig.tight_layout()
        return fig

    def plot_multipanel(self, path=None, fluence=None, mz_max=150,
                        pressure_max=1e-9, log_floor=1e-2, min_decades=2,
                        bar_color="#360f5a", dpi=300):
        """Publication multipanel: a) total outgassing, b) TEY, d) MS(t) heatmap,
        e) beam-on mass spectrum. Panel c is intentional white space.

        Built on a bare Figure with its own Agg canvas rather than through
        pyplot: the consumer calls this from several worker threads at once and
        pyplot's global figure registry is not thread-safe.
        """
        t_on, t_off = self.shutter_window()
        t_min, t_max = float(self.time[0]), float(self.time[-1])

        keep = self.mz <= mz_max
        mz_axis = [int(round(v)) for v in self.mz[keep]]
        heat = self.pressure[:, keep]
        spectrum = np.nan_to_num(self.get_spectrum()[keep])

        fig = Figure(figsize=(7, 5.5))
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(3, 2, left=0.10, right=0.80, top=0.86, bottom=0.11,
                              hspace=0.15, wspace=0.10,
                              height_ratios=[1, 1, 3.5], width_ratios=[2.2, 1])
        ax_og = fig.add_subplot(gs[0, 0])
        ax_tey = fig.add_subplot(gs[1, 0], sharex=ax_og)
        ax_heat = fig.add_subplot(gs[2, 0], sharex=ax_og)
        ax_bar = fig.add_subplot(gs[2, 1])          # right column, heatmap row only

        # ---- d: MS(t) heatmap ----------------------------------------------
        # On a bg-subtracted matrix the clip floors every negative channel to the
        # same value, so beam-off goes flat at the bottom of the colour range.
        # That is the cost of a log scale and is the intent: those negatives are
        # baseline noise about zero.
        data_log = np.log10(np.clip(heat, 1e-12, None))
        data_log = gaussian_filter1d(data_log, sigma=1, axis=0)
        clim = [i for i, m in enumerate(mz_axis) if m >= 3] or list(range(len(mz_axis)))
        im = ax_heat.imshow(data_log.T, aspect="auto", cmap="viridis",
                            vmin=np.percentile(data_log[:, clim], 1),
                            vmax=np.percentile(data_log[:, clim], 99.5),
                            interpolation="nearest",
                            extent=[t_min, t_max, len(mz_axis) - 1, 0])
        ax_heat.set_xlabel("Time (s)", fontsize=8)
        ax_heat.set_ylabel(r"$\it{m/z}$", fontsize=8)
        ax_heat.set_xticks(_nice_ticks(t_min, t_max, 5))

        mz_arr = np.array(mz_axis)
        desired = np.arange(20, mz_arr.max() + 1, 20)
        mz_ticks = sorted({int(mz_arr[np.argmin(np.abs(mz_arr - d))]) for d in desired})
        ax_heat.set_yticks([mz_axis.index(m) for m in mz_ticks])
        ax_heat.set_yticklabels([str(m) for m in mz_ticks])
        ax_heat.tick_params(length=2, pad=2)

        for ax in (ax_og, ax_tey, ax_heat):
            if t_on is not None:
                ax.axvline(t_on, color="lime", lw=0.8, ls="--", alpha=0.8)
                ax.axvline(t_off, color="red", lw=0.8, ls="--", alpha=0.8)

        def _line_panel(ax, x, y, ylabel):
            if t_on is not None:
                ax.axvspan(t_on, t_off, color="gold", alpha=0.12)
            ax.plot(x, y, color="k", lw=0.8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(length=2, pad=2, labelbottom=False)
            ax.set_xlim(t_min, t_max)
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.get_offset_text().set_fontsize(6)

        # ---- a: total outgassing -------------------------------------------
        _line_panel(ax_og, self.time, np.nansum(self.pressure, axis=1),
                    "Total\noutgas.\n(Torr)")

        # ---- b: TEY, PD-normalised when the photodiode current is known -----
        if self.pd:
            denom = abs((self.pd - (self.dark_pd or 0.0)) * 1e-6)
            tey_val, tey_label = self.tey_signal / denom, "Norm.\nTEY"
        else:
            tey_val, tey_label = self.tey_signal, "TEY\n(A)"
        _line_panel(ax_tey, self.tey_time, tey_val, tey_label)

        # ---- dose axis above panel a ----------------------------------------
        if fluence and t_on is not None:
            ax_dose = ax_og.twiny()
            ax_dose.set_xlim(ax_og.get_xlim())
            max_dose = (min(t_max, t_off) - t_on) * fluence * 1e3
            dtimes, dlabs = [], []
            for d in _nice_dose_ticks(max_dose, 5):
                tt = t_on if d == 0 else t_on + d / (fluence * 1e3)
                if t_min <= tt <= t_max:
                    dtimes.append(tt)
                    dlabs.append(f"{int(d)}" if d == int(d) else f"{d:.1f}")
            ax_dose.set_xticks(dtimes)
            ax_dose.set_xticklabels(dlabs)
            ax_dose.set_xlabel("Dose (mJ)", fontsize=7, labelpad=3)
            ax_dose.tick_params(length=2, pad=2)

        # ---- e: mass spectrum, log scale, aligned to the heatmap rows -------
        y_pos = np.arange(len(mz_axis))
        p_log = np.clip(spectrum * 1e9, log_floor, None)      # nTorr

        non_h2 = [j for j, m in enumerate(mz_axis) if m != 2]
        x_raw = (p_log[max(non_h2, key=lambda j: p_log[j])] * 2.0 if non_h2
                 else pressure_max * 1e9)
        # Snap up to the next decade: below one full decade LogFormatterSciNotation
        # starts labelling minor ticks, which smears at 7 pt in a ~1 in panel.
        x_max = max(10 ** np.ceil(np.log10(x_raw)), log_floor * 10 ** min_decades)

        # On a log axis bars must start at the floor, not 0.
        ax_bar.barh(y_pos, p_log - log_floor, left=log_floor, height=0.8,
                    color=bar_color, edgecolor=bar_color, lw=0.3, alpha=0.85)
        ax_bar.set_xscale("log")
        ax_bar.set_xlim(log_floor, x_max)
        ax_bar.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax_bar.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        ax_bar.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=16))
        ax_bar.xaxis.set_minor_formatter(NullFormatter())
        ax_bar.tick_params(axis="x", which="minor", length=1.2)
        ax_bar.set_ylim(len(mz_axis) - 1, 0)
        ax_bar.set_yticklabels([])
        ax_bar.tick_params(axis="y", length=0)
        ax_bar.set_xlabel("Pressure\n(nTorr)", fontsize=7)
        ax_bar.tick_params(axis="x", labelsize=7, length=2, pad=2)
        ax_bar.xaxis.grid(False)
        for e in range(int(np.ceil(np.log10(log_floor))),
                       int(np.floor(np.log10(x_max))) + 1):
            if log_floor < 10.0 ** e < x_max:
                ax_bar.axvline(10.0 ** e, color="gray", alpha=0.3, ls=":",
                               lw=0.5, zorder=0)
        ax_bar.set_axisbelow(True)

        cbar = fig.colorbar(im, cax=fig.add_axes([0.30, 0.02, 0.30, 0.012]),
                            orientation="horizontal")
        cbar.set_label("log(Torr, bg-sub.)" if self.is_corrected else "log(Torr)",
                       fontsize=8)
        cbar.ax.tick_params(labelsize=7, length=2, pad=2)

        # ---- panel letters (c marks the empty top-right cell) ---------------
        fig.canvas.draw()
        for ax, lab, dx in ((ax_og, "a", -0.055), (ax_tey, "b", -0.055),
                            (ax_heat, "d", -0.055), (ax_bar, "e", -0.015)):
            bb = ax.get_position()
            fig.text(bb.x0 + dx, bb.y1 + 0.015, lab, fontsize=10,
                     fontweight="bold", va="top", ha="left", family="DejaVu Sans")
        fig.text(ax_bar.get_position().x0 - 0.015,
                 ax_og.get_position().y1 + 0.015, "c", fontsize=10,
                 fontweight="bold", va="top", ha="left", family="DejaVu Sans")

        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return fig
