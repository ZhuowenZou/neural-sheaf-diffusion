"""Provenance manifest for the review bundle (server handoff 2026-09-22, section 1).

Records repository identity (SHA + dirty diff hash), environment versions, the
installed TGB package's source hashes (loader, preprocessing, evaluator,
negative sampler, label generation), raw/processed dataset file hashes,
negative-set files, split masks (hashed from the installed loader), and the
hardware.  Usage:

    python -m exp.review.provenance --out results/review_2026_09_22/provenance \
        --datasets tgbn-trade tgbn-genre tgbl-wiki thgl-forum tkgl-icews
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_array(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()


def git(args, cwd):
    try:
        return subprocess.check_output(["git"] + args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=["tgbn-trade", "tgbn-genre"])
    ap.add_argument("--skip-large-hashes", action="store_true", help="skip hashing files larger than 1 GiB")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    t0 = time.time()

    manifest = {"prepared": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "repo": repo,
                "git": {"sha": git(["rev-parse", "HEAD"], repo), "branch": git(["branch", "--show-current"], repo),
                        "status": git(["status", "--short"], repo)}}
    diff = git(["diff", "HEAD"], repo)
    manifest["git"]["dirty_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
    manifest["git"]["dirty_diff_lines"] = len(diff.splitlines())
    with open(os.path.join(args.out, "dirty_diff.patch"), "w") as fh:
        fh.write(diff)

    import torch
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    for mod in ("torch_geometric", "torch_scatter", "torch_sparse", "numpy", "pandas", "scipy", "sklearn", "torch_householder"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            env[mod] = f"unavailable ({e})"
    try:
        from importlib.metadata import version
        env["py-tgb"] = version("py-tgb")
    except Exception:
        env["py-tgb"] = "?"
    env["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    manifest["environment"] = env

    import tgb.linkproppred.dataset as ld
    tgb_root = os.path.dirname(os.path.dirname(ld.__file__))
    src_hashes = {}
    for rel in ("linkproppred/dataset.py", "linkproppred/dataset_pyg.py", "linkproppred/evaluate.py",
                "linkproppred/negative_sampler.py", "nodeproppred/dataset.py", "nodeproppred/dataset_pyg.py",
                "nodeproppred/evaluate.py", "utils/pre_process.py", "utils/utils.py", "utils/info.py",
                "datasets/dataset_scripts/tgbn-trade.py", "datasets/dataset_scripts/tgbn-genre.py"):
        p = os.path.join(tgb_root, rel)
        src_hashes[rel] = sha256_file(p) if os.path.exists(p) else "missing"
    manifest["installed_tgb"] = {"root": tgb_root, "source_sha256": src_hashes}
    # our own loader / runner sources
    ours = {}
    for rel in ("exp/temporal_benchmark_utils.py", "exp/temporal_utils.py", "exp/run_event_benchmark.py",
                "exp/run_faithful_trade.py", "exp/faithful_temporal_studies.py", "exp/temporal_mamba_studies.py",
                "models/faithful_event_model.py", "models/temporal_sheaf_ssm.py", "models/sparse_temporal_mamba.py",
                "models/laplacian_builders.py", "models/orthogonal.py", "models/sheaf_models.py"):
        ours[rel] = sha256_file(os.path.join(repo, rel))
    manifest["repo_sources_sha256"] = ours

    # hardware
    hw = {"hostname": platform.node(), "cpu_count": os.cpu_count()}
    try:
        hw["cpu_model"] = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":", 1)[1].strip()
        hw["ram_gib"] = round(int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) / 2**20, 1)
    except Exception:
        pass
    try:
        hw["gpus"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                             "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception:
        hw["gpus"] = "nvidia-smi unavailable"
    manifest["hardware"] = hw

    # datasets: files, splits, negative sets, label timestamps
    from exp import temporal_benchmark_utils as bu
    datasets = {}
    for name in args.datasets:
        spec, ds, td = bu.load_temporal_data(name)
        folder = os.path.join(tgb_root, "datasets", name.replace("-", "_"))
        files = {}
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                p = os.path.join(folder, f)
                if not os.path.isfile(p):
                    continue
                size = os.path.getsize(p)
                files[f] = {"bytes": size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))),
                            "sha256": ("skipped(>1GiB)" if args.skip_large_hashes and size > 2**30 else sha256_file(p))}
        rec = {"loader_name": spec.loader_name, "task_family": spec.task_family, "metric": spec.metric_name,
               "folder": folder, "files": files,
               "num_events": int(td.src.numel()), "num_nodes": int(max(int(td.src.max()), int(td.dst.max())) + 1),
               "t_min": int(td.t.min()), "t_max": int(td.t.max()),
               "num_rels": int(getattr(ds, "num_rels", 0) or 0),
               "src_sha256": sha256_array(td.src.numpy()), "dst_sha256": sha256_array(td.dst.numpy()),
               "t_sha256": sha256_array(td.t.numpy())}
        et = getattr(td, "edge_type", None)
        if et is not None:
            rec["edge_type_sha256"] = sha256_array(et.numpy())
        for split in ("train", "val", "test"):
            m = getattr(ds, f"{split}_mask", None)
            if m is not None:
                m = np.asarray(m)
                idx = np.nonzero(m)[0]
                rec[f"{split}_mask_sha256"] = sha256_array(m.astype(np.uint8))
                rec[f"{split}_edges"] = int(m.sum())
                rec[f"{split}_t_range"] = [int(td.t[idx].min()), int(td.t[idx].max())] if idx.size else None
        if spec.task_family == "nodeprop":
            inner = getattr(ds, "dataset", ds)
            lts = np.asarray(inner.label_ts)
            rec["label_timestamps"] = {"count": int(lts.size), "first": int(lts[0]), "last": int(lts[-1]),
                                       "sha256": sha256_array(lts), "values": lts.tolist() if lts.size <= 64 else None}
            rec["label_dict_sizes"] = {int(k): len(v) for k, v in list(inner.label_dict.items())[:64]}
        else:
            ns = {}
            for split in ("val", "test"):
                p = getattr(ds, f"{split}_ns_file", None) or getattr(getattr(ds, "dataset", ds), f"{split}_ns_file", None)
                if p is None:
                    # TGB stores <root>/<name>/<name>_<split>_ns.pkl
                    cand = os.path.join(folder, f"{spec.loader_name}_{split}_ns.pkl")
                    p = cand if os.path.exists(cand) else None
                ns[split] = {"file": p, "sha256": sha256_file(p) if p and os.path.exists(p) else "missing"}
            rec["negative_sets"] = ns
        datasets[name] = rec
        print(f"{name}: {rec['num_events']} events, {len(files)} files hashed ({time.time() - t0:.0f}s)", flush=True)
    manifest["datasets"] = datasets
    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    print(f"manifest written to {args.out}/manifest.json")


if __name__ == "__main__":
    main()
