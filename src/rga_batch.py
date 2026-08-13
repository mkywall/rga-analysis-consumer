#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rga_batch.py — standalone processing of ALS BL12.0.1.2 RGA/TEY runs.

    from rga_batch import list_samples, process_sample, process_all
    m = process_sample("TF003580")
    m.plot_summary()

Writes, per sample, into directory:
    TF003580_TEY.txt         time_s, tey_A, shutter                   [cache, raw]
    TF003580_MS_t.txt        time_s x m/z matrix of partial pressures [cache, raw]
    TF003580_MS_t_bgsub.txt  same matrix, background-subtracted       [derived]
    TF003580_RGA.txt         time_s, total_raw, total_bgsub           [derived]
    TF003580_MS.txt          mz, spectrum_raw, spectrum_bgsub         [derived]

The two [cache] files are the verbatim reparse of the instrument output and are
what process_sample() reads back on subsequent calls; never subtract into them.
_MS_t_bgsub.txt is a signed derived product, rewritten from the raw cache on
every call, and deleted if the background correction fails so that a stale
matrix can never be plotted against fresh raw data.
"""

import os
import re
import glob
import warnings
from datetime import datetime

import numpy as np

from rga_plots import PlotMixin

SUFFIXES = ("TEY", "RGA", "MS_t", "MS_t_bgsub", "MS")

_TIME_FORMATS = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S",
                 "%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S",
                 "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Low-level parsing
# ---------------------------------------------------------------------------

def _extract_float(text, pattern):
    """Float from the first capture group of *pattern* in *text*, else None."""
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1).rstrip('.'))
    except ValueError:
        return None


def _read_table(path, skiprows):
    """Tab-delimited rows as lists of str, tolerant of trailing tabs / blanks."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i < skiprows:
                continue
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            while parts and parts[-1].strip() == "":
                parts.pop()
            if parts:
                rows.append(parts)
    if not rows:
        raise ValueError(f"No data rows in {os.path.basename(path)}")
    n = min(len(r) for r in rows)
    return [r[:n] for r in rows]


def _parse_tey_file(path):
    """time_s, tey_A, shutter, pd_uA, dark_pd_uA, x, y"""
    rows = _read_table(path, skiprows=1)
    arr = np.array([[float(v) for v in r] for r in rows], dtype=float)
    time, signal = arr[:, 0], arr[:, 1]
    shutter = arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(time)

    bn = os.path.basename(path)
    return (time, signal, shutter,
            _extract_float(bn, r'_PD_([-\d.]+)uA'),
            _extract_float(bn, r'DarkPD_([-\d.]+)uA'),
            _extract_float(bn, r'_X_([-\d.]+)'),
            _extract_float(bn, r'_Y_([-\d.]+)'))


def _parse_rga_file(path):
    """time_str, mz, pressure (T,M), scan_settings, sample_name"""
    rows = _read_table(path, skiprows=2)
    time_str = [r[0] for r in rows]
    pressure = np.array([[float(v) for v in r[1:]] for r in rows], dtype=float)

    bn = os.path.basename(path)
    scan_settings = dict(
        scanspeed=_extract_float(bn, r'scanspeed_(\d+)'),
        finalmass=_extract_float(bn, r'finalmass_(\d+)'),
        scantime=_extract_float(bn, r'scantime_(\d+)'),
    )
    mz = np.arange(1, pressure.shape[1] + 1, dtype=float)
    fm = scan_settings.get("finalmass")
    if fm and abs(fm - pressure.shape[1]) > 1:
        warnings.warn(f"{bn}: finalmass={fm:g} but {pressure.shape[1]} channels; "
                      f"assuming m/z = 1..{pressure.shape[1]}", stacklevel=2)

    sample_name = bn.split('_RGA_')[0] if '_RGA_' in bn else None
    return time_str, mz, pressure, scan_settings, sample_name


def _timestamps_to_seconds(time_str_list, dt_fallback=1.0):
    """Timestamp strings -> seconds relative to the first scan."""
    for fmt in _TIME_FORMATS:
        try:
            t0 = datetime.strptime(time_str_list[0], fmt)
            return np.array([(datetime.strptime(t, fmt) - t0).total_seconds()
                             for t in time_str_list])
        except ValueError:
            continue
    try:                                    # already numeric?
        return np.array([float(t) for t in time_str_list])
    except ValueError:
        pass
    warnings.warn(f"Unrecognised timestamp {time_str_list[0]!r}; "
                  f"assuming uniform {dt_fallback} s spacing.", stacklevel=2)
    return np.arange(len(time_str_list), dtype=float) * dt_fallback


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def out_paths(sample, outdir):
    return {s: os.path.join(outdir, f"{sample}_{s}.txt") for s in SUFFIXES}


def match_sample_files(sample, directory):
    """All raw TEY and RGA files whose name begins exactly with '{sample}_'.

    The sample name is by convention everything before '_TEY_'/'_RGA_', so the
    prefix is anchored: a glob of '{sample}*_TEY_' would let TF0035 swallow
    TF003580's files.
    """
    mine = {os.path.basename(p) for p in out_paths(sample, directory).values()}

    def _hits(pattern):
        return [h for h in sorted(glob.glob(os.path.join(directory, pattern)))
                if os.path.isfile(h) and os.path.basename(h) not in mine]

    return _hits(f"{sample}_TEY_DarkPD_*"), _hits(f"{sample}_RGA_histogram_*")


def find_sample_files(sample, directory):
    """The one raw TEY and one raw RGA file for *sample* (RGA may lack .txt)."""
    tey_hits, rga_hits = match_sample_files(sample, directory)

    def _pick(hits, kind):
        if not hits:
            raise FileNotFoundError(
                f"No raw {kind} file for {sample!r} in {directory}")
        if len(hits) > 1:
            warnings.warn(f"{len(hits)} {kind} files match {sample!r}; using "
                          f"{os.path.basename(hits[0])}", stacklevel=3)
        return hits[0]

    return _pick(tey_hits, "TEY"), _pick(rga_hits, "RGA")


def list_samples(directory):
    """Sample names that have an RGA histogram file in *directory*."""
    return sorted({os.path.basename(f).split("_RGA_")[0]
                   for f in glob.glob(os.path.join(directory, "*_RGA_histogram_*"))
                   if os.path.isfile(f)})


# ---------------------------------------------------------------------------
# Header / metadata I/O
# ---------------------------------------------------------------------------

def _meta_block(meta):
    return "\n".join(f"{k}: {'' if v is None else v}" for k, v in meta.items())


def _read_meta(path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line.lstrip("#").strip()
            if ":" in body:
                k, v = body.split(":", 1)
                v = v.strip()
                try:
                    v = float(v)
                except ValueError:
                    v = v or None
                meta[k.strip()] = v
    return meta


def _column_header_line(path, startswith):
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line.lstrip("#").strip()
            if body.startswith(startswith):
                return body.split("\t")
    return None


def _write_tey(path, t, sig, shut, meta):
    np.savetxt(path, np.column_stack([t, sig, shut]),
               fmt=("%.6f", "%.9e", "%g"), delimiter="\t",
               header=_meta_block(meta) + "\ntime_s\ttey_A\tshutter", comments="# ")


def _write_ms_t(path, t, mz, p, meta):
    cols = "\t".join(f"{v:g}" for v in mz)
    np.savetxt(path, np.column_stack([t, p]), fmt="%.9e", delimiter="\t",
               header=_meta_block(meta) + f"\ntime_s\t{cols}", comments="# ")


def _write_ms_t_bgsub(path, t, mz, p, meta):
    """Signed background-subtracted matrix. Derived, not a cache — do not
    feed this back into background_correct()."""
    cols = "\t".join(f"{v:g}" for v in mz)
    np.savetxt(path, np.column_stack([t, p]), fmt="%.9e", delimiter="\t",
               header=_meta_block(meta) +
                      "\nBACKGROUND-SUBTRACTED matrix - signed - NOT the raw cache"
                      f"\ntime_s\t{cols}", comments="# ")


def _write_rga(path, t, raw, cor, meta):
    if cor is None:
        arr, names = np.column_stack([t, raw]), "time_s\ttotal_raw_Torr"
    else:
        arr = np.column_stack([t, raw, cor])
        names = "time_s\ttotal_raw_Torr\ttotal_bgsub_Torr"
    np.savetxt(path, arr, fmt="%.9e", delimiter="\t",
               header=_meta_block(meta) + "\n" + names, comments="# ")


def _write_ms(path, mz, raw, cor, meta):
    if cor is None:
        arr, names, fmt = np.column_stack([mz, raw]), "mz\tpressure_raw_Torr", ("%g", "%.9e")
    else:
        arr = np.column_stack([mz, raw, cor])
        names = "mz\tpressure_raw_Torr\tpressure_bgsub_Torr"
        fmt = ("%g", "%.9e", "%.9e")
    np.savetxt(path, arr, fmt=fmt, delimiter="\t",
               header=_meta_block(meta) + "\n" + names, comments="# ")


def _read_tey(path):
    d = np.loadtxt(path, comments="#", delimiter="\t", ndmin=2)
    return d[:, 0], d[:, 1], d[:, 2], _read_meta(path)


def _read_ms_t(path):
    """Works for both _MS_t.txt and _MS_t_bgsub.txt (identical layout)."""
    d = np.loadtxt(path, comments="#", delimiter="\t", ndmin=2)
    names = _column_header_line(path, "time_s")
    mz = (np.array([float(v) for v in names[1:]]) if names
          else np.arange(1, d.shape[1], dtype=float))
    return d[:, 0], mz, d[:, 1:], _read_meta(path)


# ---------------------------------------------------------------------------
# Measurement object
# ---------------------------------------------------------------------------

class RGAMeasurement(PlotMixin):
    """One RGA + TEY run: pressures (T, M), TEY time series, and metadata.

    The plot_* methods come from rga_plots.PlotMixin.
    """

    def __init__(self, sample_name, time, mz, pressure,
                 tey_time, tey_signal, shutter, meta=None):
        self.sample_name = sample_name
        self.time        = np.asarray(time, float)
        self.mz          = np.asarray(mz, float)
        self.pressure    = np.asarray(pressure, float)
        self.tey_time    = np.asarray(tey_time, float)
        self.tey_signal  = np.asarray(tey_signal, float)
        self.shutter     = np.asarray(shutter, float)
        self.meta        = dict(meta or {})
        self.pd          = self.meta.get("pd_uA")
        self.dark_pd     = self.meta.get("dark_pd_uA")
        self.x           = self.meta.get("x")
        self.y           = self.meta.get("y")
        self.chamber_pressure = self.meta.get("chamber_pressure_Torr")
        self.scan_settings = {k: self.meta.get(k)
                              for k in ("scanspeed", "finalmass", "scantime")}

    def __repr__(self):
        bg = "bg-corrected" if hasattr(self, "_raw_pressure") else "raw"
        return (f"<RGAMeasurement {self.sample_name} "
                f"{self.n_timepoints}x{self.n_mz} {bg}>")

    @property
    def n_timepoints(self):
        return len(self.time)

    @property
    def n_mz(self):
        return len(self.mz)

    @property
    def is_corrected(self):
        return hasattr(self, "_raw_pressure")

    def get_trace(self, mz_val):
        """Pressure vs time for one m/z channel."""
        idx = int(np.argmin(np.abs(self.mz - mz_val)))
        if abs(self.mz[idx] - mz_val) > 0.5:
            raise ValueError(f"m/z {mz_val} not in dataset")
        return self.pressure[:, idx]

    def get_spectrum(self, t=None, t_start=None, t_end=None):
        """Mass spectrum, optionally averaged over a time window."""
        if t is not None:
            return self.pressure[t, :]
        if t_start is not None or t_end is not None:
            lo = self.time[0] if t_start is None else t_start
            hi = self.time[-1] if t_end is None else t_end
            mask = (self.time >= lo) & (self.time <= hi)
            if not mask.any():
                raise ValueError(f"No RGA scans in [{lo:.1f}, {hi:.1f}] s.")
            return np.nanmean(self.pressure[mask, :], axis=0)
        if hasattr(self, "open_time"):
            mask = (self.time >= self.open_time) & (self.time <= self.close_time)
            if mask.any():
                return np.nanmean(self.pressure[mask, :], axis=0)
        return np.nanmean(self.pressure, axis=0)

    def top_masses(self, n=15, mz_min=1):
        """(m/z, intensity) pairs for the n strongest peaks in the spectrum."""
        spec = self.get_spectrum()
        sel = self.mz >= mz_min
        order = np.argsort(spec[sel])[::-1][:n]
        return list(zip(self.mz[sel][order], spec[sel][order]))

    # -- background --------------------------------------------------------

    def background_correct(self, window=30.0, gap_before=5.0, gap_after=10.0):
        """Per-channel linear baseline from two beam-off windows (in place).

        Falls back to a flat offset (deg=0) if only one of the two windows
        contains scans; the degree actually used is recorded in self.bg_deg.
        The baseline is evaluated over the whole time axis, so any data beyond
        the post-close window is extrapolated.
        """
        edges = np.diff((self.shutter > 0.5).astype(int))
        open_idx, close_idx = np.where(edges > 0)[0], np.where(edges < 0)[0]
        if len(open_idx) == 0 or len(close_idx) == 0:
            raise ValueError("Could not detect shutter open/close edges.")

        open_time  = self.tey_time[open_idx[0] + 1]
        close_time = self.tey_time[close_idx[-1]]

        off1_end   = open_time - gap_before
        off1_start = off1_end - window
        off1 = (self.time >= off1_start) & (self.time <= off1_end)

        off2_start = close_time + gap_after
        off2_end   = off2_start + window
        off2 = (self.time >= off2_start) & (self.time <= off2_end)

        n1, n2 = int(off1.sum()), int(off2.sum())
        if n1 + n2 < 2:
            raise ValueError(
                f"Not enough background points (before={n1}, after={n2}). "
                f"Increase 'window' or reduce 'gap_before'/'gap_after'.")
        if n1 == 0:
            warnings.warn(f"No RGA scans in pre-shutter window "
                          f"[{off1_start:.1f}, {off1_end:.1f}] s; "
                          f"falling back to a flat offset.", stacklevel=2)
        if n2 == 0:
            warnings.warn(f"No RGA scans in post-shutter window "
                          f"[{off2_start:.1f}, {off2_end:.1f}] s; "
                          f"falling back to a flat offset.", stacklevel=2)

        bg = off1 | off2
        x_bg = self.time[bg]
        deg = 1 if (n1 and n2) else 0          # flat offset if only one window

        if not hasattr(self, "_raw_pressure"):
            self._raw_pressure = self.pressure.copy()
        src = self._raw_pressure
        cor = np.empty_like(src)
        for j in range(src.shape[1]):
            col = src[:, j]
            coeffs = np.polyfit(x_bg, col[bg], deg)
            cor[:, j] = col - np.polyval(coeffs, self.time)

        self.pressure    = cor
        self.open_time   = open_time
        self.close_time  = close_time
        self.bg_deg      = deg
        self.bg_n_before = n1
        self.bg_n_after  = n2
        self._bg_off1    = (off1_start, off1_end)
        self._bg_off2    = (off2_start, off2_end)
        return self

    def compute_outgas_area(self, mz_range=None):
        """Integrated outgassing over the beam-on window, in Torr*s.

        Sums the background-corrected channels, then trapezoid-integrates
        between shutter open and close with both boundary values interpolated
        at the exact times. Negatives are clipped to zero first, so baseline
        noise about zero cannot subtract from the area.

        Returns None if the background correction did not run: an area off an
        uncorrected trace would just be the chamber baseline times the window.
        """
        if not self.is_corrected:
            return None

        sel = (np.ones(self.n_mz, bool) if mz_range is None
               else (self.mz >= mz_range[0]) & (self.mz <= mz_range[1]))
        total = np.nansum(self.pressure[:, sel], axis=1)

        # Clamp to the RGA time axis before interpolating. np.interp would
        # otherwise silently extend the endpoint value out to the shutter time
        # and report a full-width area over a trace that never covered it.
        lo = max(float(self.open_time), float(self.time[0]))
        hi = min(float(self.close_time), float(self.time[-1]))
        if hi <= lo:
            warnings.warn(
                f"[{self.sample_name}] beam-on window "
                f"{self.open_time:.1f}-{self.close_time:.1f} s does not overlap "
                f"the RGA trace {self.time[0]:.1f}-{self.time[-1]:.1f} s; "
                f"no outgassing area.", stacklevel=2)
            return None
        if lo > self.open_time or hi < self.close_time:
            warnings.warn(
                f"[{self.sample_name}] beam-on window "
                f"{self.open_time:.1f}-{self.close_time:.1f} s extends past the "
                f"RGA trace; area integrated over {lo:.1f}-{hi:.1f} s only.",
                stacklevel=2)

        inner = (self.time > lo) & (self.time < hi)
        t_roi = np.concatenate([[lo], self.time[inner], [hi]])
        p_roi = np.concatenate([[np.interp(lo, self.time, total)],
                                total[inner],
                                [np.interp(hi, self.time, total)]]).clip(min=0)
        return float(np.trapezoid(p_roi, t_roi))

    def reset(self):
        """Undo background correction."""
        if hasattr(self, "_raw_pressure"):
            self.pressure = self._raw_pressure.copy()
            for a in ("_raw_pressure", "open_time", "close_time", "bg_deg",
                      "bg_n_before", "bg_n_after", "_bg_off1", "_bg_off2"):
                if hasattr(self, a):
                    delattr(self, a)
        return self

    def shutter_window(self):
        """(open, close) from the shutter trace, or self.open/close_time if set."""
        if self.is_corrected:
            return float(self.open_time), float(self.close_time)
        on = self.shutter > 0.5
        if not on.any():
            return None, None
        e = np.diff(on.astype(int))
        up, dn = np.where(e > 0)[0], np.where(e < 0)[0]
        return (float(self.tey_time[up[0] + 1] if len(up) else self.tey_time[on][0]),
                float(self.tey_time[dn[-1]] if len(dn) else self.tey_time[on][-1]))


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def process_sample(sample, directory, outdir=None, overwrite=False,
                   background=True, window=30.0, gap_before=5.0, gap_after=10.0,
                   t_offset=0.0, mz_range=None, save_plots=True, verbose=True):
    """
    Process one sample end to end, write the output files, return the measurement.

    overwrite  : re-parse the raw instrument files even if the cache exists.
    t_offset   : seconds added to the TEY axis to align it with the RGA axis.
    mz_range   : (lo, hi) to restrict the summed-pressure trace in the _RGA file.
    save_plots : also write {sample}_multipanel.png.

    Five files are written when the background correction succeeds, four when it
    does not — in the latter case any pre-existing _MS_t_bgsub.txt is deleted so
    that downstream plots cannot mix baselines from different runs.

    The returned measurement carries, for callers that upload the results:
        .outgas_area  Torr*s over the beam-on window, or None if uncorrected
        .metadata     flat dict of every run condition and analysis parameter
        .files        raw TEY, raw RGA, then every derived file written
        .thumbnail    the PNG path, or None
    """
    outdir = directory if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    paths = out_paths(sample, outdir)

    cached = (not overwrite and os.path.exists(paths["TEY"])
              and os.path.exists(paths["MS_t"]))

    if cached:
        tey_time, tey_signal, shutter, meta = _read_tey(paths["TEY"])
        time, mz, pressure, meta_ms = _read_ms_t(paths["MS_t"])
        meta = {**meta_ms, **meta}
        if verbose:
            print(f"[{sample}] cache -> {sample}_TEY.txt, {sample}_MS_t.txt")
    else:
        tey_raw, rga_raw = find_sample_files(sample, directory)
        tey_time, tey_signal, shutter, pd_ua, dark_pd_ua, x, y = _parse_tey_file(tey_raw)
        time_str, mz, pressure, scan, _ = _parse_rga_file(rga_raw)
        time = _timestamps_to_seconds(time_str)

        tey_bn = os.path.basename(tey_raw)
        meta = dict(
            sample=sample, pd_uA=pd_ua, dark_pd_uA=dark_pd_ua,
            chamber_pressure_Torr=_extract_float(tey_bn, r"_Pressure_([-\d.eE+]+)Torr"),
            x=x, y=y,
            scanspeed=scan.get("scanspeed"), finalmass=scan.get("finalmass"),
            scantime=scan.get("scantime"), t0=time_str[0],
            tey_source=tey_bn, rga_source=os.path.basename(rga_raw),
        )
        _write_tey(paths["TEY"], tey_time, tey_signal, shutter, meta)
        _write_ms_t(paths["MS_t"], time, mz, pressure, meta)
        if verbose:
            print(f"[{sample}] parsed raw -> wrote {sample}_TEY.txt, {sample}_MS_t.txt")

    m = RGAMeasurement(sample, time, mz, pressure,
                       np.asarray(tey_time) + t_offset, tey_signal, shutter, meta)

    corrected = False
    if background:
        try:
            m.background_correct(window=window, gap_before=gap_before,
                                 gap_after=gap_after)
            corrected = True
            if verbose:
                kind = "linear" if m.bg_deg == 1 else "flat offset"
                print(f"[{sample}] background corrected ({kind}, "
                      f"{m.bg_n_before}+{m.bg_n_after} scans); beam on "
                      f"{m.open_time:.1f}–{m.close_time:.1f} s")
        except Exception as e:
            warnings.warn(f"[{sample}] background correction failed: {e}", stacklevel=2)

    raw_p = getattr(m, "_raw_pressure", m.pressure)
    sel = (np.ones(m.n_mz, bool) if mz_range is None
           else (m.mz >= mz_range[0]) & (m.mz <= mz_range[1]))

    total_raw = np.nansum(raw_p[:, sel], axis=1)
    total_cor = np.nansum(m.pressure[:, sel], axis=1) if corrected else None

    tmask = np.ones(m.n_timepoints, bool)
    if corrected:
        w = (m.time >= m.open_time) & (m.time <= m.close_time)
        if w.any():
            tmask = w
    spec_raw = np.nanmean(raw_p[tmask, :], axis=0)
    spec_cor = np.nanmean(m.pressure[tmask, :], axis=0) if corrected else None

    area = m.compute_outgas_area(mz_range=mz_range)

    dmeta = dict(meta)
    dmeta.update(background_corrected=corrected, bg_window_s=window,
                 bg_gap_before_s=gap_before, bg_gap_after_s=gap_after,
                 bg_fit_deg=(m.bg_deg if corrected else None),
                 bg_n_scans_before=(m.bg_n_before if corrected else None),
                 bg_n_scans_after=(m.bg_n_after if corrected else None),
                 t_offset_s=t_offset,
                 mz_sum_range=("all" if mz_range is None
                               else f"{mz_range[0]}-{mz_range[1]}"),
                 beam_on_window_s=(f"{m.open_time:.1f}-{m.close_time:.1f}"
                                   if corrected else None),
                 spectrum_window_s=(f"{m.open_time:.1f}-{m.close_time:.1f}"
                                    if corrected else "all"),
                 outgas_area_Torr_s=area)

    _write_rga(paths["RGA"], m.time, total_raw, total_cor, dmeta)
    _write_ms(paths["MS"], m.mz, spec_raw, spec_cor, dmeta)
    if corrected:
        _write_ms_t_bgsub(paths["MS_t_bgsub"], m.time, m.mz, m.pressure, dmeta)
    elif os.path.exists(paths["MS_t_bgsub"]):
        os.remove(paths["MS_t_bgsub"])   # stale matrix from an earlier good run
        if verbose:
            print(f"[{sample}] removed stale {sample}_MS_t_bgsub.txt")
    if verbose:
        print(f"[{sample}] wrote {sample}_RGA.txt, {sample}_MS.txt"
              + (f", {sample}_MS_t_bgsub.txt" if corrected else ""))

    m.total_pressure = total_cor if corrected else total_raw
    m.spectrum = spec_cor if corrected else spec_raw
    m.outgas_area = area
    m.metadata = {k: (v.item() if isinstance(v, np.generic) else v)
                  for k, v in dmeta.items()}

    m.thumbnail = None
    if save_plots:
        plot_path = os.path.join(outdir, f"{sample}_multipanel.png")
        pdf_path = os.path.join(outdir, f"{sample}_multipanel.pdf")
        try:
            for output in [plot_path, pdf_path]:
                m.plot_multipanel(path=output)
            m.thumbnail = plot_path
            if verbose:
                print(f"[{sample}] wrote {sample}_multipanel.png and {sample}_multipanel.pdf")
        except Exception as e:
            # A figure failure must not cost us the data files.
            warnings.warn(f"[{sample}] plotting failed: {e}", stacklevel=2)

    # Raw first: the consumer timestamps the dataset off a raw file, and the
    # derived files were written seconds ago.
    m.files = [os.path.join(directory, meta["tey_source"]),
               os.path.join(directory, meta["rga_source"])]
    m.files += [p for p in paths.values() if os.path.exists(p)]
    if save_plots:
        m.files.append(pdf_path)
    return m


def process_all(directory, **kwargs):
    """Process every sample in *directory*; returns {sample: measurement}."""
    out = {}
    for s in list_samples(directory):
        try:
            out[s] = process_sample(s, directory=directory, **kwargs)
        except Exception as e:
            warnings.warn(f"{s}: {e}", stacklevel=2)
    return out


if __name__ == "__main__":
    import sys
    directory = sys.argv[1]
    for name, meas in process_all(directory).items():
        flag = "" if meas.is_corrected else "  [NOT bg-corrected]"
        print(f"{name}: {meas.n_timepoints} scans x {meas.n_mz} channels{flag}")