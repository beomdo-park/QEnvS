import contextlib
import os
import pathlib
import re
import sys


def _limit_cpus() -> None:
    """Pin this test process to a few cores before jax starts its thread pool.

    XLA sizes its CPU thread pool from the process affinity, and its thread flags have no
    effect on recent jax, so one test process can use every core of a shared machine. Each
    pytest-xdist worker gets its own QENVS_TEST_CPUS cores (default 2); a run without xdist
    gets 4. QENVS_TEST_CPUS=all disables the limit.

    Affinity is per thread on Linux, and an xdist worker runs this file in a side thread
    after the main thread has already started a BLAS pool, so every existing thread is
    pinned, not only the caller; threads started later inherit the mask.
    """
    setting = os.environ.get("QENVS_TEST_CPUS", "")
    if setting == "all" or not hasattr(os, "sched_setaffinity"):
        return
    worker = re.fullmatch(r"gw(\d+)", os.environ.get("PYTEST_XDIST_WORKER", ""))
    if not worker and any(a == "-n" or a.startswith(("-n", "--numprocesses")) for a in sys.argv):
        return  # the xdist controller is idle; pinning it would shrink the workers' cores
    cpus = sorted(os.sched_getaffinity(0))
    per = int(setting) if setting else (2 if worker else 4)
    start = (int(worker.group(1)) * per) % len(cpus) if worker else 0
    mask = [cpus[(start + i) % len(cpus)] for i in range(min(per, len(cpus)))]
    for task in pathlib.Path("/proc/self/task").iterdir():
        with contextlib.suppress(ProcessLookupError, PermissionError):  # thread exited meanwhile
            os.sched_setaffinity(int(task.name), mask)


_limit_cpus()

import jax  # noqa: E402 (after the CPU limit above)

jax.config.update("jax_enable_x64", True)
