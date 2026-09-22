"""Generate launch command files (.cmd) for the review campaign.

Each .cmd execs results/event_bench/queues/wait_launch.sh (GPU policy) with the
full command; the monitor daemon relaunches crashed ones (<= 4 attempts).

  python -m exp.review.make_queue wiki-wave1          # 8 arms x seed 43 x lr {3e-4, 1e-3}
  python -m exp.review.make_queue wiki-wave2 --lr-json <file>   # seeds 44-47 with the chosen lr per arm
  python -m exp.review.make_queue audits <run> ...    # checkpoint-replay audits of old runs
"""
import argparse
import json
import os
import stat

ROOT = "/home/zhuowez1/project/neural-sheaf-diffusion"
RV = "results/review_2026_09_22"
PY = "/home/zhuowez1/miniconda3/envs/nsd/bin/python"
WL = "results/event_bench/queues/wait_launch.sh"

WIKI_BASE = ("--dataset tgbl-wiki --model faithful --time-window 600 --track-val-edges 8000 --train-negatives-per-pos 32 "
             "--epochs 8 --patience 5 --min-epochs 4 --recurrency-decoder --predict-from-previous --save-checkpoint --rng-isolation --dump-query-ranks")
ARMS = {
    "tsd": "",
    "curonly": "--sheaf-conditioning current_only",
    "identity": "--spatial identity",
    "gru": "--backbone gru --spatial identity",
    "diag": "--backbone diag_ssm --spatial identity",
    "attention": "--spatial attention",
    "nodeframe": "--spatial node_frame",
    "coreoff": "--no-memory --layers 0 --fast-core-off",
}
SKIP_KEYS = {"out", "eval_only", "save_checkpoint", "skip_final_eval", "reserve_gpu_mb", "epochs", "patience", "min_epochs",
             "rng_isolation", "no_query_audit", "clock_diagnostics", "audit_eval_only", "backbone", "spatial", "clock",
             "fast_core_off", "delta_time_scale"}


def write_cmd(name, min_mib, cmd, only_gpus=None, subdir="queue"):
    d = os.path.join(ROOT, RV, subdir)
    os.makedirs(d, exist_ok=True)
    log = f"{RV}/{subdir}/{name}.log"
    p = os.path.join(d, f"{name}.cmd")
    env = f"export TSD_ONLY_GPUS=\"{only_gpus}\"\n" if only_gpus else ""
    with open(p, "w") as fh:
        fh.write(f"#!/bin/bash\ncd {ROOT}\n{env}exec {WL} {min_mib} {log} {cmd} > {log} 2>&1\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return p


def config_to_flags(cfg):
    parts = []
    for k, v in sorted(cfg.items()):
        if k in SKIP_KEYS or v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
        else:
            parts.append(f"{flag} {v}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["wiki-wave1", "wiki-wave2", "wiki-extras", "audits", "synth"])
    ap.add_argument("--gen", default="gen_s1", help="synthetic generator directory under results/review_2026_09_22/synthetic")
    ap.add_argument("--seeds", default="43 44 45 46 47")
    ap.add_argument("--rec", default="off", choices=["off", "on"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--lr-json", default=None)
    ap.add_argument("--gpus", default="3 4 7")
    ap.add_argument("--audit-gpus", default="0")
    args = ap.parse_args()
    gpus = args.gpus.split()
    written = []
    if args.what == "wiki-wave1":
        i = 0
        for arm, extra in ARMS.items():
            for lr in ("3e-4", "1e-3"):
                name = f"wiki_{arm}_s43_lr{lr}"
                out = f"{RV}/matched/{name}"
                cmd = f"{PY} -m exp.run_event_benchmark {WIKI_BASE} {extra} --lr {lr} --seed 43 --out {out}"
                written.append(write_cmd(name, 2000, cmd, only_gpus=gpus[i % len(gpus)])); i += 1
    elif args.what == "wiki-wave2":
        lrs = json.load(open(args.lr_json))
        i = 0
        for arm, extra in ARMS.items():
            lr = lrs[arm]
            for seed in (44, 45, 46, 47):
                name = f"wiki_{arm}_s{seed}_lr{lr}"
                out = f"{RV}/matched/{name}"
                cmd = f"{PY} -m exp.run_event_benchmark {WIKI_BASE} {extra} --lr {lr} --seed {seed} --out {out}"
                written.append(write_cmd(name, 2000, cmd, only_gpus=gpus[i % len(gpus)])); i += 1
    elif args.what == "wiki-extras":
        # clock comparison (node clocks, same core/head/batching) and batching diagnostic (0.5x / 2x width),
        # TSD at seed 43 with the locked lr; fixed-membership gap intervention is a separate script
        lrs = json.load(open(args.lr_json)); lr = lrs["tsd"]; i = 0
        for name, extra in (("wiki_tsd_clock-node_update_s43", "--clock node_update --clock-diagnostics"),
                            ("wiki_tsd_clock-node_interaction_s43", "--clock node_interaction --clock-diagnostics"),
                            ("wiki_tsd_clock-global_s43", "--clock-diagnostics"),
                            ("wiki_tsd_tw300_s43", "--clock-diagnostics"), ("wiki_tsd_tw1200_s43", "--clock-diagnostics")):
            base = WIKI_BASE.replace("--time-window 600", "--time-window 300" if "tw300" in name else ("--time-window 1200" if "tw1200" in name else "--time-window 600"))
            out = f"{RV}/clock/{name}"
            cmd = f"{PY} -m exp.run_event_benchmark {base} {extra} --lr {lr} --seed 43 --out {out}"
            written.append(write_cmd(name, 2000, cmd, only_gpus=gpus[i % len(gpus)])); i += 1
    elif args.what == "synth":
        SY = (f"--dataset synth-history:{RV}/synthetic/{args.gen}/data.npz --model faithful --time-window 20000 --context-edges 2000 "
              "--track-val-edges 3000 --train-negatives-per-pos 32 --epochs 6 --patience 3 --min-epochs 3 --lr 1e-3 "
              "--predict-from-previous --save-checkpoint --rng-isolation --dump-query-ranks")
        rec = " --recurrency-decoder" if args.rec == "on" else ""
        i = 0
        for arm, extra in ARMS.items():
            for seed in args.seeds.split():
                name = f"synth_{args.gen}_{arm}_rec{args.rec}_s{seed}"
                out = f"{RV}/synthetic/runs/{name}"
                cmd = f"{PY} -m exp.run_event_benchmark {SY}{rec} {extra} --seed {seed} --out {out}"
                written.append(write_cmd(name, 1500, cmd, only_gpus=gpus[i % len(gpus)])); i += 1
    else:
        import pandas as pd
        for run in args.runs:
            src = f"results/event_bench/leakfree2/{run}"
            r = pd.read_csv(os.path.join(ROOT, src, "results.csv")).iloc[0]
            cfg = json.loads(r["config_json"])
            ckpt = os.path.join(src, "best.pt")
            if not os.path.exists(os.path.join(ROOT, ckpt)):
                # icews: training run saved best.pt, evaluation run lives in <name>_eval_sXX
                alt = src.replace("_eval_", "_f_")
                ckpt = os.path.join(alt, "best.pt")
            assert os.path.exists(os.path.join(ROOT, ckpt)), ckpt
            flags = config_to_flags(cfg)
            name = f"audit_{run}"
            out = f"{RV}/audit/{run}"
            cmd = (f"{PY} -m exp.run_event_benchmark {flags} --epochs 0 --eval-only {ckpt} --audit-eval-only --clock-diagnostics "
                   f"--predict-from-previous --out {out}")
            mem = {"tkgl-icews": 14000, "tkgl-polecat": 12000, "tkgl-wikidata": 11000, "thgl-forum": 7000,
                   "tkgl-smallpedia": 22000, "thgl-software": 5000, "tgbl-wiki": 2000}.get(cfg["dataset"], 8000)
            written.append(write_cmd(name, mem, cmd, only_gpus=args.audit_gpus, subdir="queue"))
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
