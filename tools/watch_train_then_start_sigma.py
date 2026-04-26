#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WatchTarget:
    pid: int
    cmdline: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ""
    parts = [part for part in raw.decode(errors="ignore").split("\x00") if part]
    return " ".join(parts)


def _read_ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split(":", 1)[1].strip())
    except FileNotFoundError:
        return None
    return None


def _pid_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _matching_targets(substring: str) -> list[WatchTarget]:
    matches: list[WatchTarget] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        cmdline = _read_cmdline(pid)
        if substring in cmdline:
            matches.append(WatchTarget(pid=pid, cmdline=cmdline))
    return matches


def _select_root_target(matches: list[WatchTarget]) -> WatchTarget:
    if not matches:
        raise RuntimeError("No matching process found.")
    matched_pids = {target.pid for target in matches}
    roots = [target for target in matches if _read_ppid(target.pid) not in matched_pids]
    candidates = roots or matches
    return min(candidates, key=lambda target: target.pid)


def _resolve_watch_target(*, pid: int | None, watch_substring: str) -> WatchTarget:
    if pid is not None:
        cmdline = _read_cmdline(pid)
        if not cmdline:
            raise RuntimeError(f"PID {pid} does not exist.")
        return WatchTarget(pid=pid, cmdline=cmdline)
    return _select_root_target(_matching_targets(watch_substring))


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _append_log(path: Path, message: str) -> None:
    _ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_timestamp()}] {message}\n")


def _wait_for_pid_exit(
    *,
    target: WatchTarget,
    poll_seconds: float,
    log_path: Path,
) -> None:
    _append_log(
        log_path,
        f"watching pid={target.pid} cmd={target.cmdline!r}",
    )
    while _pid_exists(target.pid):
        time.sleep(poll_seconds)
    _append_log(log_path, f"watched pid={target.pid} exited")


def _wait_for_file(
    *,
    path: Path,
    timeout_seconds: float,
    poll_seconds: float,
    log_path: Path,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.is_file():
            _append_log(log_path, f"resolved checkpoint {path}")
            return
        time.sleep(poll_seconds)
    raise RuntimeError(f"Timed out waiting for checkpoint: {path}")


def _command_running(substring: str) -> bool:
    return bool(_matching_targets(substring))


def _launch_sigma_training(
    *,
    repo_root: Path,
    sigma_run: str,
    field: str,
    mu_model_path: Path,
    train_log_path: Path,
    log_path: Path,
) -> int:
    dragon_exe = shutil.which("dragon")
    if dragon_exe is None:
        raise RuntimeError("Could not resolve 'dragon' on PATH.")

    command = [
        dragon_exe,
        "train",
        "-n",
        sigma_run,
        "-t",
        "sigma",
        "-f",
        field,
        "--mu-model",
        str(mu_model_path),
    ]
    _ensure_parent_dir(train_log_path)
    train_log = train_log_path.open("a", encoding="utf-8")
    train_log.write(f"[{_timestamp()}] launch {' '.join(shlex.quote(part) for part in command)}\n")
    train_log.flush()
    child = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=train_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _append_log(
        log_path,
        f"launched sigma pid={child.pid} cmd={' '.join(shlex.quote(part) for part in command)}",
    )
    return child.pid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch a running mu training process, then launch sigma training.",
    )
    parser.add_argument("--pid", type=int, default=None, help="PID of the current mu training root process.")
    parser.add_argument(
        "--watch-substring",
        default="dragon train -n ret_mu -t mu -f ret",
        help="Fallback process substring when --pid is omitted.",
    )
    parser.add_argument("--field", default="ret", help="Field passed to the follow-up sigma training.")
    parser.add_argument("--mu-run", default="ret_mu", help="Checkpoint run name for the frozen mu model.")
    parser.add_argument("--sigma-run", default="ret_sigma", help="Run name for the follow-up sigma training.")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval while waiting.")
    parser.add_argument(
        "--checkpoint-wait-seconds",
        type=float,
        default=600.0,
        help="How long to wait for the mu best checkpoint after the watched process exits.",
    )
    parser.add_argument(
        "--log-file",
        default="models/automation_logs/watch_ret_mu_to_ret_sigma.log",
        help="Watchdog log file.",
    )
    parser.add_argument(
        "--train-log-file",
        default="models/automation_logs/ret_sigma.log",
        help="stdout/stderr log file for the launched sigma training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and log the handoff plan without launching sigma training.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    log_path = (repo_root / args.log_file).resolve()
    train_log_path = (repo_root / args.train_log_file).resolve()
    mu_model_path = (repo_root / "models" / "checkpoints" / args.mu_run / "best.pt").resolve()
    sigma_substring = f"dragon train -n {args.sigma_run} -t sigma -f {args.field}"

    try:
        target = _resolve_watch_target(pid=args.pid, watch_substring=args.watch_substring)
        _wait_for_pid_exit(target=target, poll_seconds=args.poll_seconds, log_path=log_path)
        _wait_for_file(
            path=mu_model_path,
            timeout_seconds=args.checkpoint_wait_seconds,
            poll_seconds=args.poll_seconds,
            log_path=log_path,
        )
        if _command_running(sigma_substring):
            _append_log(log_path, f"skip launch because sigma command already running: {sigma_substring!r}")
            return 0
        if args.dry_run:
            _append_log(
                log_path,
                f"dry-run ready: dragon train -n {args.sigma_run} -t sigma -f {args.field} --mu-model {mu_model_path}",
            )
            return 0
        _launch_sigma_training(
            repo_root=repo_root,
            sigma_run=args.sigma_run,
            field=args.field,
            mu_model_path=mu_model_path,
            train_log_path=train_log_path,
            log_path=log_path,
        )
        return 0
    except Exception as exc:
        _append_log(log_path, f"fatal: {exc}")
        print(f"watchdog failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
