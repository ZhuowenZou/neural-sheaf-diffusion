"""Bounded clock / saturation diagnostics for the event model (review handoff
2026-09-22, section 2).

The owning event model calls `begin_step(...)` with per-local-node metadata
before the memory update; the SSM's step selector then calls `record_steps(...)`
with the per-node selector pre-activation, time term, uncapped and capped step
and the gap actually supplied.  Exact counters are kept per (split, activity
class); a reservoir sample of at most `reservoir` rows keeps the joint
distribution for quantiles.  Nothing here changes the model's computation."""
import math
import os

import numpy as np
import pandas as pd
import torch


class ClockDiagnostics:
    FIELDS = ["split", "activity", "t_batch", "t_prev_batch", "t_last_update", "t_last_interaction",
              "seen_update", "seen_interaction", "gap_used", "gap_global", "gap_since_update",
              "gap_since_interaction", "delta_scale", "selector_preactivation", "time_term",
              "dt_uncapped", "dt", "cap_active"]

    def __init__(self, reservoir=200_000, seed=0):
        self.reservoir = int(reservoir)
        self.rng = np.random.default_rng(seed)
        self.split = "unlabelled"
        self.counts = {}
        self.sample = []
        self.n_seen_total = 0
        self._meta = None

    def _c(self, split, activity):
        key = (split, activity)
        if key not in self.counts:
            self.counts[key] = dict(n=0, n_capped=0, n_zero_gap_used=0, n_zero_gap_global=0, n_first_update=0,
                                    n_first_interaction=0, sum_dt=0.0, sum_dt_uncapped=0.0, sum_log10_gap_used=0.0,
                                    n_gap_used_pos=0, hist_dt=np.zeros(12, dtype=np.int64),
                                    hist_log10_gap_update=np.zeros(14, dtype=np.int64),
                                    hist_log10_gap_interaction=np.zeros(14, dtype=np.int64))
        return self.counts[key]

    def begin_step(self, t_batch, t_prev_batch, last_update, last_interaction, seen_update,
                   seen_interaction, is_endpoint):
        """All per-node tensors are aligned with the local node set of this step."""
        self._meta = dict(t_batch=float(t_batch), t_prev_batch=float(t_prev_batch) if t_prev_batch is not None else float("nan"),
                          last_update=last_update, last_interaction=last_interaction, seen_update=seen_update,
                          seen_interaction=seen_interaction, is_endpoint=is_endpoint)

    @staticmethod
    def _hist_dt(dt, cap):
        # 12 bins on [0, cap]: last bin = exactly at the cap
        edges = np.linspace(0.0, cap, 12) if np.isfinite(cap) else np.linspace(0.0, 1.0, 12)
        idx = np.clip(np.searchsorted(edges, dt, side="right") - 1, 0, 11)
        return np.bincount(idx, minlength=12)

    @staticmethod
    def _hist_log10(gap):
        # bins: [0], (0,1e-2], ..., decades up to 1e12
        out = np.zeros(14, dtype=np.int64)
        out[0] = int((gap <= 0).sum())
        pos = gap[gap > 0]
        if pos.size:
            idx = np.clip(np.floor(np.log10(pos)).astype(np.int64) + 3, 1, 13)
            out += np.bincount(idx, minlength=14)
        return out

    @torch.no_grad()
    def record_steps(self, content, time_term, dt_uncapped, dt, gap_used, delta_scale, cap):
        m = self._meta
        n = dt.numel()
        if m is None:
            m = dict(t_batch=float("nan"), t_prev_batch=float("nan"), last_update=None, last_interaction=None,
                     seen_update=None, seen_interaction=None, is_endpoint=None)
        # chunked SSM calls: metadata rows [offset, offset+n)
        off = getattr(self, "_offset", 0)
        sl = slice(off, off + n)
        self._offset = off + n

        def take(t, default):
            if t is None:
                return np.full(n, default)
            return t[sl].detach().cpu().numpy()

        dt_np = dt.cpu().numpy().astype(np.float64)
        dtu_np = dt_uncapped.cpu().numpy().astype(np.float64)
        gap_np = gap_used.cpu().numpy().astype(np.float64)
        pre_np = content.cpu().numpy().astype(np.float64)
        tt_np = time_term.cpu().numpy().astype(np.float64)
        endpoint = take(m["is_endpoint"], True).astype(bool)
        lu = take(m["last_update"], float("nan")).astype(np.float64)
        li = take(m["last_interaction"], float("nan")).astype(np.float64)
        su = take(m["seen_update"], False).astype(bool)
        si = take(m["seen_interaction"], False).astype(bool)
        gap_global = m["t_batch"] - m["t_prev_batch"] if np.isfinite(m["t_prev_batch"]) else float("nan")
        g_upd = np.where(su, m["t_batch"] - lu, np.nan)
        g_int = np.where(si, m["t_batch"] - li, np.nan)
        capped = dt_np >= (cap - 1e-7) if np.isfinite(cap) else np.zeros(n, dtype=bool)
        for act, mask in (("endpoint", endpoint), ("closure", ~endpoint)):
            k = int(mask.sum())
            if not k:
                continue
            c = self._c(self.split, act)
            c["n"] += k
            c["n_capped"] += int(capped[mask].sum())
            c["n_zero_gap_used"] += int((gap_np[mask] <= 0).sum())
            c["n_zero_gap_global"] += k if (np.isfinite(gap_global) and gap_global <= 0) else 0
            c["n_first_update"] += int((~su[mask]).sum())
            c["n_first_interaction"] += int((~si[mask]).sum())
            c["sum_dt"] += float(dt_np[mask].sum())
            c["sum_dt_uncapped"] += float(dtu_np[mask].sum())
            pos = gap_np[mask] > 0
            c["n_gap_used_pos"] += int(pos.sum())
            c["sum_log10_gap_used"] += float(np.log10(gap_np[mask][pos]).sum()) if pos.any() else 0.0
            c["hist_dt"] += self._hist_dt(dt_np[mask], cap)
            c["hist_log10_gap_update"] += self._hist_log10(np.nan_to_num(g_upd[mask], nan=-1.0))
            c["hist_log10_gap_interaction"] += self._hist_log10(np.nan_to_num(g_int[mask], nan=-1.0))
        # reservoir sample (Algorithm R over all rows seen so far)
        rows = np.stack([np.full(n, m["t_batch"]), np.full(n, m["t_prev_batch"]), lu, li, su, si, gap_np,
                         np.full(n, gap_global), g_upd, g_int, np.full(n, delta_scale), pre_np, tt_np, dtu_np, dt_np,
                         capped], axis=1)
        for i in range(n):
            self.n_seen_total += 1
            if len(self.sample) < self.reservoir:
                self.sample.append((self.split, "endpoint" if endpoint[i] else "closure", rows[i]))
            else:
                j = int(self.rng.integers(0, self.n_seen_total))
                if j < self.reservoir:
                    self.sample[j] = (self.split, "endpoint" if endpoint[i] else "closure", rows[i])

    def end_step(self):
        self._offset = 0
        self._meta = None

    def frames(self):
        agg = []
        for (split, act), c in self.counts.items():
            n = max(c["n"], 1)
            agg.append(dict(split=split, activity=act, n=c["n"], n_capped=c["n_capped"], frac_capped=c["n_capped"] / n,
                            n_zero_gap_used=c["n_zero_gap_used"], frac_zero_gap_used=c["n_zero_gap_used"] / n,
                            n_zero_gap_global=c["n_zero_gap_global"], n_first_update=c["n_first_update"],
                            n_first_interaction=c["n_first_interaction"], mean_dt=c["sum_dt"] / n,
                            mean_dt_uncapped=c["sum_dt_uncapped"] / n,
                            mean_log10_gap_used_pos=(c["sum_log10_gap_used"] / c["n_gap_used_pos"]) if c["n_gap_used_pos"] else float("nan"),
                            hist_dt=" ".join(map(str, c["hist_dt"].tolist())),
                            hist_log10_gap_update=" ".join(map(str, c["hist_log10_gap_update"].tolist())),
                            hist_log10_gap_interaction=" ".join(map(str, c["hist_log10_gap_interaction"].tolist())),
                            reservoir_rows=len(self.sample), rows_seen=self.n_seen_total))
        samp = pd.DataFrame([[s, a] + list(r) for s, a, r in self.sample], columns=self.FIELDS)
        return pd.DataFrame(agg), samp

    def write(self, out_dir, prefix="clock"):
        os.makedirs(out_dir, exist_ok=True)
        agg, samp = self.frames()
        agg.to_csv(os.path.join(out_dir, f"{prefix}_diagnostics.csv"), index=False)
        samp.to_csv(os.path.join(out_dir, f"{prefix}_reservoir.csv.gz"), index=False, compression="gzip")
        # quantile summary from the reservoir per (split, activity)
        rows = []
        if len(samp):
            for (split, act), g in samp.groupby(["split", "activity"]):
                q = lambda col: np.nanquantile(g[col].astype(float), [0.05, 0.25, 0.5, 0.75, 0.95]).round(6).tolist()
                rows.append(dict(split=split, activity=act, n_sample=len(g), dt_q=q("dt"), dt_uncapped_q=q("dt_uncapped"),
                                 gap_used_q=q("gap_used"), gap_since_update_q=q("gap_since_update"),
                                 gap_since_interaction_q=q("gap_since_interaction"), preact_q=q("selector_preactivation"),
                                 time_term_q=q("time_term")))
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{prefix}_quantiles.csv"), index=False)
        return agg
