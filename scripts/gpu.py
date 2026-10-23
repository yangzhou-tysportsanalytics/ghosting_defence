"""Learned ghost on a GPU cluster in short batch jobs (D-022): every step is resumable.

The same steps as ``scripts/phase3_gpu.sh``, cut so that a job of any length makes progress and
the next job continues where it stopped (training stops at an epoch boundary before the job's
time limit; the deviations stop between games). State lives in ``runs/`` and the data directory,
so a later job may run on another node or another GPU of the same shared file system.

Usage (in the bundle directory; the environment comes from ``uv sync --locked``):
    .venv/bin/python scripts/gpu.py status                  # what is done, what comes next
    .venv/bin/python scripts/gpu.py next                    # run pending steps until time is up
    .venv/bin/python scripts/gpu.py choose --suggested      # after the sweep: fix the config
    .venv/bin/python scripts/gpu.py run crossfit:fold3      # one step (parallel jobs)
    .venv/bin/python scripts/gpu.py pack                    # ghost_gpu_results.zip (no coordinates)
    .venv/bin/python scripts/gpu.py selftest                # ~5 min: every code path on the GPU
Time limit: ``--time-limit auto`` (default) reads the remaining time of the Slurm job
(``squeue``); outside Slurm there is no limit unless given (``2:50:00``, ``170m``, ``3h``).
Exit codes of next / run: 0 finished, 3 time is up (submit the job again), 4 waiting for
``choose``, 5 no progress in this job (time limit too short, or other jobs hold the steps),
1 failure (see reports/logs/gpu_<step>.log).
``--dry-run``: the whole chain on a CPU in minutes (tags ``dry_*``; numbers meaningless).

Selection uses the validation games only; ``choose`` refuses to change the config once the test
games have been scored (``final``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = {  # config suffix -> model / optimiser arguments (as in phase3_gpu.sh)
    "d64_lr3e-4": ["--d-model", "64", "--blocks", "2", "--n-heads", "4", "--lr", "3e-4"],
    "d64_lr1e-3": ["--d-model", "64", "--blocks", "2", "--n-heads", "4", "--lr", "1e-3"],
    "d128_lr3e-4": ["--d-model", "128", "--blocks", "4", "--n-heads", "8", "--lr", "3e-4"],
    "d128_lr1e-3": ["--d-model", "128", "--blocks", "4", "--n-heads", "8", "--lr", "1e-3"],
}
FOLDS = range(5)
EXIT_DONE, EXIT_FAIL, EXIT_TIME, EXIT_CHOOSE, EXIT_STALL = 0, 1, 3, 4, 5


@dataclass
class Mode:
    prefix: str  # "gpu" (real) or "dry"
    device: str
    epochs: int
    extra: list = field(default_factory=list)  # sweep / final / ablation / conditions
    cf: list = field(default_factory=list)  # cross-fitting
    dev: list = field(default_factory=list)  # deviations
    cal: list = field(default_factory=list)  # calibration


def make_mode(dry: bool, device: str | None) -> Mode:
    if dry:  # three epochs, so that a run can be cut and resumed
        return Mode("dry", device or "cpu", 3,
                    ["--train-games", "2", "--val-games", "2", "--max-batches", "3",
                     "--max-eval", "64"],
                    ["--max-batches", "3", "--max-eval", "64", "--epochs", "1"], ["--games", "1"],
                    ["--games", "1"])  # fmt: skip
    return Mode("gpu", device or "cuda", 30)


def data_paths() -> tuple[str, Path]:
    from ghost import data as D

    cfg = D.DataConfig.load()
    return cfg.version, cfg.processed_dir / "all"


# ---------------------------------------------------------------------------- state and locks
def state_dir() -> Path:
    d = ROOT / "runs" / "gpu_state"
    (d / "locks").mkdir(parents=True, exist_ok=True)
    return d


def journal(msg: str) -> None:
    line = f"{time.strftime('%F %T')} [{os.environ.get('SLURM_JOB_ID', 'local')}] {msg}"
    print(line, flush=True)
    with open(state_dir() / "journal.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def chosen(mode: Mode) -> str | None:
    p = state_dir() / f"{mode.prefix}_config.txt"
    return p.read_text().strip() if p.exists() else None


def _lock_path(step: str) -> Path:
    return state_dir() / "locks" / (step.replace(":", "_") + ".lock")


def lock_holder(step: str) -> dict | None:
    """The live lock of a step, or None (a lock past its job's end is stale)."""
    p = _lock_path(step)
    try:
        info = json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return info if info.get("until", 0) > time.time() else None


def acquire(step: str, until: float) -> bool:
    p = _lock_path(step)
    if lock_holder(step) is not None:
        return False
    p.unlink(missing_ok=True)  # stale
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:  # another job won the race
        return False
    info = {"job": os.environ.get("SLURM_JOB_ID", "local"), "host": socket.gethostname(),
            "pid": os.getpid(), "until": until}  # fmt: skip
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    return True


def release(step: str) -> None:
    _lock_path(step).unlink(missing_ok=True)


# ---------------------------------------------------------------------------------- the steps
@dataclass
class Step:
    name: str
    kind: str  # check | train | calibration | deviations
    tag: str = ""
    args: list = field(default_factory=list)
    deps: list = field(default_factory=list)
    needs_config: bool = False
    min_minutes: float = 0  # real runs: time needed to start a step that cannot be cut


def steps(mode: Mode) -> list[Step]:
    p = mode.prefix
    out = [Step("check", "check"),
           Step("smoke", "train", f"{p}_smoke", ["--epochs", "1", "--train-games", "2",
                                                  "--val-games", "2", *mode.extra[4:]],
                ["check"])]  # fmt: skip
    for s, a in SWEEP.items():
        out.append(Step(f"sweep:{s}", "train", f"{p}_{s}", [*a, *mode.extra], ["check"]))
    c = chosen(mode)
    if c is None:
        return out + [Step("choose", "gate", needs_config=True)]
    ca = SWEEP[c.removeprefix(f"{p}_")]
    version, _ = data_paths()
    sweeps = [f"sweep:{s}" for s in SWEEP]
    out += [
        Step("final", "train", f"{c}_final",
             [*ca, "--eval-test", "--load-model", f"runs/phase3/{version}_all_{c}/model.pt",
              *mode.extra], sweeps, min_minutes=20),
        Step("calibration", "calibration", f"{c}_final", list(mode.cal), ["final"],
             min_minutes=30),
        Step("ablation:noanchor", "train", f"{c}_noanchor", [*ca, "--anchor", "none", *mode.extra],
             ["final"]),
        Step("ablation:l2", "train", f"{c}_l2", [*ca, "--loss", "l2", *mode.extra], ["final"]),
        Step("conditions:lineup", "train", f"{c}_lineup", [*ca, "--condition", "lineup",
             *mode.extra], ["final"]),
        Step("conditions:scheme", "train", f"{c}_scheme", [*ca, "--condition", "scheme",
             *mode.extra], ["final"]),
    ]  # fmt: skip
    folds = [f"crossfit:fold{k}" for k in FOLDS]
    for k in FOLDS:
        out.append(Step(folds[k], "train", f"{c}_fold{k}", [*ca, "--fold-out", str(k), *mode.cf],
                        ["final"]))  # fmt: skip
    out.append(Step("deviations", "deviations", c, list(mode.dev), folds))
    return out


def eval_json(tag: str) -> Path:
    version, _ = data_paths()
    return ROOT / "reports" / "phase3" / f"{version}_all_{tag}_eval.json"


def is_done(step: Step, mode: Mode) -> bool:
    version, all_dir = data_paths()
    if step.kind == "check":
        return (state_dir() / f"{mode.prefix}_check.ok").exists()
    if step.kind == "gate":
        return chosen(mode) is not None
    if step.kind == "train":
        return eval_json(step.tag).exists()
    if step.kind == "calibration":
        return (ROOT / "reports" / "phase3"
                / f"{version}_all_{step.tag}_calibration_by_context.json").exists()  # fmt: skip
    if step.kind == "deviations":
        return (ROOT / "reports" / "phase4" / f"{version}_all"
                / f"learned_ghost_dev_cf_{step.tag}.json").exists()  # fmt: skip
    raise ValueError(step.kind)


def progress(step: Step) -> tuple:
    """Something that changes whenever a job advances the step (for the no-progress guard)."""
    version, all_dir = data_paths()
    if step.kind == "train":
        ck = ROOT / "runs" / "phase3" / f"{version}_all_{step.tag}" / "checkpoint.pt"
        return (ck.stat().st_mtime if ck.exists() else None,)
    if step.kind == "deviations":
        d = all_dir / "learned_ghost" / f"cf_{step.tag}"
        return (len(list(d.glob("*.parquet"))) if d.exists() else 0,)
    return ()


def train_epoch(step: Step) -> str:
    """'k/E epochs' from the checkpoint of an unfinished training run."""
    version, _ = data_paths()
    ck = ROOT / "runs" / "phase3" / f"{version}_all_{step.tag}" / "checkpoint.pt"
    if not ck.exists():
        return ""
    import torch

    st = torch.load(ck, map_location="cpu", weights_only=False)
    return f"{st['epoch'] + 1}/{st['tcfg']['epochs']} epochs"


# ------------------------------------------------------------------------------ running steps
def run_check(mode: Mode) -> int:
    _, all_dir = data_paths()
    root = all_dir / "ghost_arrays"
    man = json.loads((root / "manifest.json").read_text())
    for split, info in man["splits"].items():
        for f, want in info["files"].items():
            h = hashlib.sha256()
            with open(root / split / f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != want:
                journal(f"FAILED check: checksum mismatch {split}/{f}")
                return EXIT_FAIL
        journal(f"check: {split} {info['n_possessions']} possessions, checksums ok")
    (state_dir() / f"{mode.prefix}_check.ok").write_text(time.strftime("%F %T"))
    return EXIT_DONE


def command(step: Step, mode: Mode, deadline: float | None) -> list[str]:
    _, all_dir = data_paths()
    arr = (all_dir / "ghost_arrays").as_posix()
    dl = ["--deadline", f"{deadline:.0f}"] if deadline else []
    py = [sys.executable]
    if step.kind == "train":
        epochs = [] if "--epochs" in step.args else ["--epochs", str(mode.epochs)]
        return py + ["scripts/phase3_train.py", "--game-set", "all", "--arrays-dir", arr,
                     "--device", mode.device, "--schedule", "mixed", "--loss", "nll",
                     "--anchor", "rule", "--batch-size", "64", *epochs, "--tag", step.tag,
                     "--resume", *dl, *step.args]  # fmt: skip
    if step.kind == "calibration":
        return py + ["scripts/phase3_calibration_by_context.py", "--model-run", step.tag,
                     "--split", "test", "--device", mode.device, *step.args]  # fmt: skip
    if step.kind == "deviations":
        return py + ["scripts/phase4_learned_ghost_dev.py", "--crossfit", step.tag,
                     "--arrays-dir", arr, "--device", mode.device, "--resume", *dl,
                     *step.args]  # fmt: skip
    raise ValueError(step.kind)


def execute(step: Step, mode: Mode, deadline: float | None) -> int:
    if step.kind == "check":
        return run_check(mode)
    log = ROOT / "reports" / "logs" / f"gpu_{step.name.replace(':', '_')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(step, mode, deadline)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"=== {time.strftime('%F %T')} job {os.environ.get('SLURM_JOB_ID', 'local')} "
                f"{' '.join(cmd[1:])}\n")  # fmt: skip
        f.flush()
        env = {**os.environ, "PYTHONUTF8": "1"}
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc not in (EXIT_DONE, EXIT_TIME):
        journal(f"FAILED {step.name} (exit {rc}); see {log.relative_to(ROOT).as_posix()}")
        return EXIT_FAIL
    return rc


# ------------------------------------------------------------------------------------- timing
def parse_duration(s: str) -> float:
    """'2:50:00', '1-02:00:00', '45:00' (MM:SS, squeue style), '170m', '3h', '90s' -> seconds."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", s)
    if m:
        return float(m[1]) * {"s": 1, "m": 60, "h": 3600}[m[2]]
    days, _, rest = s.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, mi, se = parts
    return (int(days or 0) * 24 + h) * 3600 + mi * 60 + se


def job_deadline(limit: str, margin_s: float) -> tuple[float | None, float]:
    """(deadline for the steps, end of the job for the locks)."""
    now = time.time()
    if limit == "none":
        return None, now + 7 * 86400
    if limit == "auto":
        jid = os.environ.get("SLURM_JOB_ID")
        if not jid:
            return None, now + 7 * 86400
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"], capture_output=True,
                             text=True).stdout.strip()  # fmt: skip
        if not out or out.upper() in ("UNLIMITED", "NOT_SET", "INVALID"):
            sys.exit(f"cannot read the remaining time of job {jid} ({out!r}); use --time-limit")
        remaining = parse_duration(out)
    else:
        remaining = parse_duration(limit)
    return now + remaining - margin_s, now + remaining


def gpu_ok(mode: Mode) -> bool:
    if not mode.device.startswith("cuda"):
        return True
    import torch

    if not torch.cuda.is_available():
        journal("FAILED: torch does not see a GPU on this node")
        return False
    journal(f"node {socket.gethostname()}, GPU {torch.cuda.get_device_name(0)}, "
            f"torch {torch.__version__} (CUDA {torch.version.cuda})")  # fmt: skip
    try:  # create a context: a faulty GPU / MIG slice is visible but fails here
        float((torch.ones(8, device="cuda") * 2).sum())
    except Exception as e:  # noqa: BLE001
        node = socket.gethostname().split(".")[0]
        journal(f"FAILED: the GPU on {node} is unusable ({type(e).__name__}: {str(e)[:120]}); "
                f"resubmit with sbatch --exclude={node} ...")  # fmt: skip
        return False
    return True


# ----------------------------------------------------------------------------------- commands
def run_steps(
    mode: Mode, todo: list[Step], deadline: float | None, until: float, min_start: float = 120
) -> int:
    advanced, held = False, []
    for st in todo:
        if is_done(st, mode):
            continue
        if st.kind == "gate":
            journal("waiting for `choose`: run `scripts/gpu.py choose --suggested` (see status)")
            return EXIT_CHOOSE
        if not all(is_done(d, mode) for d in steps(mode) if d.name in st.deps):
            if len(todo) == 1:
                journal(f"{st.name} waits for {', '.join(st.deps)}")
            continue  # its inputs are still being produced (by another job)
        need = max(min_start, 60 * st.min_minutes if mode.prefix == "gpu" else 0)
        if deadline is not None and deadline - time.time() < need:
            journal(f"less than {need / 60:.1f} min left; {st.name} starts in the next job")
            break
        if not acquire(st.name, until):
            held.append(st.name)
            continue
        try:
            before = progress(st)
            journal(f"start {st.name}")
            rc = execute(st, mode, deadline)
            changed = progress(st) != before
        finally:
            release(st.name)
        if rc == EXIT_FAIL:
            return EXIT_FAIL
        advanced = advanced or rc == EXIT_DONE or changed
        if rc == EXIT_TIME:
            journal(f"time is up during {st.name}; submit the job again to continue")
            return EXIT_TIME if advanced else EXIT_STALL
        journal(f"done {st.name}")
    remaining = [s.name for s in steps(mode) if not is_done(s, mode)]
    if not remaining:
        return EXIT_DONE
    if held:
        journal(f"held by other jobs: {', '.join(held)}")
    return EXIT_TIME if advanced else EXIT_STALL


def cmd_next(mode: Mode, args) -> int:
    deadline, until = job_deadline(args.time_limit, args.margin * 60)
    if not gpu_ok(mode):
        return EXIT_FAIL
    if deadline:
        journal(f"time for steps: {(deadline - time.time()) / 60:.0f} min")
    rc = run_steps(mode, steps(mode), deadline, until, args.min_start * 60)
    if rc == EXIT_DONE:
        journal("all steps done")
        cmd_pack(mode)
    return rc


def cmd_run(mode: Mode, args) -> int:
    st = {s.name: s for s in steps(mode)}.get(args.step)
    if st is None:
        sys.exit(f"unknown step {args.step}; see `status` (steps after `choose` need a config)")
    deadline, until = job_deadline(args.time_limit, args.margin * 60)
    if not gpu_ok(mode):
        return EXIT_FAIL
    if is_done(st, mode):
        return EXIT_DONE
    rc = run_steps(mode, [st], deadline, until, args.min_start * 60)
    # run_steps judges the whole pipeline; here only this step counts
    return EXIT_DONE if rc != EXIT_FAIL and is_done(st, mode) else rc


def sweep_table(mode: Mode) -> list[tuple[float, str]]:
    rows = []
    for s in SWEEP:
        tag = f"{mode.prefix}_{s}"
        p = eval_json(tag)
        if not p.exists():
            print(f"  {tag}: not finished")
            continue
        e = json.loads(p.read_text())
        t, i = e["val_team"]["slot_mixture_nll_ft"], e["val_individual"]["nll_ft"]
        rows.append(((t + i) / 2, tag))
        print(f"  {tag}: val team NLL {t:.4f}  individual NLL {i:.4f}  mean {(t + i) / 2:.4f}")
    return rows


def cmd_choose(mode: Mode, args) -> int:
    print("validation NLL of the sweep (lower is better):")
    rows = sweep_table(mode)
    if len(rows) < len(SWEEP) and not args.force:
        sys.exit("the sweep is not finished (use --force to choose anyway)")
    if args.suggested:
        tag = min(rows)[1]
    elif args.config:
        tag = args.config if args.config.startswith(f"{mode.prefix}_") else (
            f"{mode.prefix}_{args.config}")  # fmt: skip
        if tag.removeprefix(f"{mode.prefix}_") not in SWEEP or not eval_json(tag).exists():
            sys.exit(f"{tag}: not a finished sweep run")
    else:
        if rows:
            print("suggested:", min(rows)[1])
        return EXIT_DONE
    old = chosen(mode)
    if old and old != tag and eval_json(f"{old}_final").exists() and not args.force:
        sys.exit(f"{old} has already been scored on the test games; the choice stays")
    (state_dir() / f"{mode.prefix}_config.txt").write_text(tag + "\n")
    journal(f"config chosen: {tag}")
    return EXIT_DONE


def cmd_status(mode: Mode, args) -> int:
    c = chosen(mode)
    print(f"mode {mode.prefix} (device {mode.device}); config: {c or 'not chosen yet'}")
    nxt, running = None, None
    for st in steps(mode):
        if is_done(st, mode):
            state = "done"
        elif st.kind == "gate":
            sweep_done = all(is_done(s, mode) for s in steps(mode) if s.name.startswith("sweep:"))
            state = ("waiting: run `scripts/gpu.py choose --suggested`" if sweep_done
                     else "after the sweep")  # fmt: skip
        elif (h := lock_holder(st.name)) is not None:
            state = f"running in job {h['job']} on {h['host']}"
            running = running or (st.name, h["job"])
        else:
            part = train_epoch(st) if st.kind == "train" else ""
            state = f"started ({part})" if part else "not started"
        nxt = nxt or (None if state == "done" or state.startswith("running") else st.name)
        print(f"  {st.name:<20} {state}")
    if running:
        print(f"\nrunning: {running[0]} in job {running[1]}; nothing to submit until that job "
              "ends (with CHAIN=1 the next job is submitted automatically)")  # fmt: skip
    elif nxt is None:
        z = ROOT / "ghost_gpu_results.zip"
        print("all done;", f"{z.name} ready" if z.exists() else "run `scripts/gpu.py pack`")
    elif nxt == "choose":
        print("\nnext: choose the config (validation only):")
        sweep_table(mode)
    else:
        print(f"\nnext: {nxt} (submit the job again, or run `scripts/gpu.py next`)")
    return EXIT_DONE


def cmd_selftest(mode: Mode, args) -> int:
    """Every kind of step on a few batches on this device (training, resume, deadline stop, test
    scoring, conditioned ghost, cross-fitting, calibration, deviations); outputs deleted."""
    if not gpu_ok(mode):
        return EXIT_FAIL
    import shutil

    version, all_dir = data_paths()
    arr = (all_dir / "ghost_arrays").as_posix()
    tiny = ["--max-batches", "2", "--max-eval", "64"]
    sub = ["--train-games", "2", "--val-games", "2"]
    train = [sys.executable, "scripts/phase3_train.py", "--game-set", "all", "--arrays-dir", arr,
             "--device", mode.device, "--schedule", "mixed", "--loss", "nll", "--anchor", "rule",
             "--batch-size", "64", "--resume"]  # fmt: skip
    run_dir = f"runs/phase3/{version}_all_selftest"
    checks = [  # name, command, expected exit code, text expected in the output
        ("train 1 epoch", train + ["--tag", "selftest", "--epochs", "1", *sub, *tiny], 0, ""),
        ("resume from checkpoint", train + ["--tag", "selftest", "--epochs", "2", *sub, *tiny],
         0, "resumed_from_epoch"),
        ("stop at deadline", train + ["--tag", "selftest", "--epochs", "3", "--deadline",
         f"{time.time():.0f}", *sub, *tiny], EXIT_TIME, "INCOMPLETE"),
        ("score test (final)", train + ["--tag", "selftest_final", "--eval-test", "--load-model",
         f"{run_dir}/model.pt", *sub, *tiny], 0, ""),
        ("lineup ghost", train + ["--tag", "selftest_lineup", "--epochs", "1", "--condition",
         "lineup", *sub, *tiny], 0, ""),
        ("cross-fitting fold", train + ["--tag", "selftest_fold0", "--epochs", "1", "--fold-out",
         "0", *tiny], 0, ""),
        ("calibration by context", [sys.executable, "scripts/phase3_calibration_by_context.py",
         "--model-run", "selftest_final", "--split", "test", "--device", mode.device,
         "--games", "1"], 0, ""),
        ("deviations", [sys.executable, "scripts/phase4_learned_ghost_dev.py", "--model-run",
         "selftest", "--arrays-dir", arr, "--device", mode.device, "--games", "1", "--resume"],
         0, ""),
    ]  # fmt: skip
    log = ROOT / "reports" / "logs" / "gpu_selftest.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    ok = True
    with open(log, "w", encoding="utf-8") as f:
        for name, cmd, want, text in checks:
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, "PYTHONUTF8": "1"})  # fmt: skip
            out = r.stdout + r.stderr
            f.write(f"=== {name}: {' '.join(cmd[1:])}\n{out}\n")
            good = r.returncode == want and text in out
            ok = ok and good
            print(f"  {'PASS' if good else 'FAIL'}  {name} ({time.time() - t0:.0f} s)", flush=True)
            if not good:
                print("\n".join(out.strip().splitlines()[-8:]))
    for p in [*ROOT.glob("runs/phase3/*_selftest*"), *ROOT.glob("reports/phase3/*_selftest*"),
              *ROOT.glob("reports/phase4/*/*selftest*"), *all_dir.glob("learned_ghost/selftest*"),
              *all_dir.glob("analysis/*selftest*")]:  # fmt: skip
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    journal(f"selftest {'passed' if ok else 'FAILED'} (log: reports/logs/gpu_selftest.log)")
    return EXIT_DONE if ok else EXIT_FAIL


def cmd_pack(mode: Mode, args=None) -> int:
    _, all_dir = data_paths()
    items = [
        ROOT / "runs" / "phase3",
        ROOT / "runs" / "gpu_state",
        ROOT / "reports" / "phase3",
        ROOT / "reports" / "phase4",
        ROOT / "reports" / "logs",
        all_dir / "learned_ghost",
    ]
    items += sorted((all_dir / "analysis").glob("learned_ghost_dev_*.parquet"))  # fmt: skip
    items = [it if it.is_absolute() else ROOT / it for it in items]  # data dir may be relative
    z = ROOT / "ghost_gpu_results.zip"
    tmp = z.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as f:
        for it in items:
            files = [it] if it.is_file() else (sorted(it.rglob("*")) if it.exists() else [])
            for p in files:  # checkpoints (optimiser state) and locks are not needed back
                if p.is_file() and p.name != "checkpoint.pt" and p.suffix != ".lock":
                    f.write(p, arcname=p.resolve().relative_to(ROOT).as_posix())
    tmp.replace(z)  # an interrupted pack leaves no partial zip
    journal(f"packed {z.name}: {z.stat().st_size:,} bytes (no coordinates)")
    return EXIT_DONE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="CPU rehearsal (tags dry_*)")
    ap.add_argument("--device", default=None, help="default cuda (cpu with --dry-run)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("next", "run"):
        sp = sub.add_parser(name)
        if name == "run":
            sp.add_argument("step", help="e.g. sweep:d64_lr3e-4, final, crossfit:fold2")
        sp.add_argument("--time-limit", default="auto",
                        help="auto (Slurm job) | none | 2:50:00 | 170m | 3h")  # fmt: skip
        sp.add_argument("--margin", type=float, default=8.0,
                        help="minutes kept free at the end (final evaluation, saving)")  # fmt: skip
        sp.add_argument("--min-start", type=float, default=2.0,
                        help="minutes needed to start another step")  # fmt: skip
    ch = sub.add_parser("choose")
    ch.add_argument("config", nargs="?", help="sweep config, e.g. d64_lr1e-3")
    ch.add_argument("--suggested", action="store_true", help="lowest mean validation NLL")
    ch.add_argument("--force", action="store_true")
    sub.add_parser("status")
    sub.add_parser("selftest")
    sub.add_parser("pack")
    args = ap.parse_args()
    os.chdir(ROOT)
    mode = make_mode(args.dry_run, args.device)
    fn = {"next": cmd_next, "run": cmd_run, "choose": cmd_choose, "status": cmd_status,
          "pack": cmd_pack, "selftest": cmd_selftest}[args.cmd]  # fmt: skip
    return fn(mode, args)


if __name__ == "__main__":
    sys.exit(main())
