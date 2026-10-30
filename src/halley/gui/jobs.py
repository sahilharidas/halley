"""The GUI's jobs: builds and runs, one at a time on a thread of their own.

A build traces a model's formulas and a run projects every policy; either can take minutes (a
first Numba build compiles every kernel), so the API starts one as a :class:`Job` and the page
asks after it until it is done. One thread runs the jobs in the order they were submitted. The
engine is not written for two builds or runs at once in one process: tracing a model sets up
symbolic state and a build captures warnings, both of which belong to the process, and two runs
at once would compete for the same cores. A job cannot be stopped once it has started, since the
engine has no way to stop a run part way; a queued job can be.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from halley.gui.catalog import exception_name, exception_text

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass(slots=True)
class Job:
    """One build or run: what it is, where it stands and, once finished, its result or error."""

    id: str
    kind: str
    label: str
    state: str = QUEUED
    submitted: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: object = None
    error: dict[str, object] | None = None
    future: Future[None] | None = field(default=None, repr=False)
    on_cancel: Callable[[], None] | None = field(default=None, repr=False)
    """Called once when the job is cancelled before it starts, to undo what its submission
    reserved (a run directory's name, say)."""

    @property
    def active(self) -> bool:
        return self.state in (QUEUED, RUNNING)

    def as_dict(self) -> dict[str, object]:
        """The job as the page reads it: the seconds it has run, or ran, beside its state."""
        end = self.finished if self.finished is not None else time.time()
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "submitted": self.submitted,
            "seconds": None if self.started is None else round(end - self.started, 3),
            "result": self.result if self.state == DONE else None,
            "error": self.error,
        }


class Jobs:
    """The queue of jobs and the one thread that runs them."""

    def __init__(self, keep: int = 200) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._ids = itertools.count(1)
        self._keep = keep
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="halley-gui-job")

    def submit(
        self,
        kind: str,
        label: str,
        work: Callable[[], object],
        describe_error: Callable[[BaseException], dict[str, object]],
        on_cancel: Callable[[], None] | None = None,
    ) -> Job:
        """Queue ``work``. Its return value becomes the job's result; an exception it raises
        becomes the job's error, as ``describe_error`` describes it. ``on_cancel`` is called if
        the job is cancelled before it starts."""
        with self._lock:
            job = Job(f"job-{next(self._ids)}", kind, label, on_cancel=on_cancel)
            self._jobs[job.id] = job
            self._forget_old()
        job.future = self._executor.submit(self._run, job, work, describe_error)
        return job

    def _run(
        self,
        job: Job,
        work: Callable[[], object],
        describe_error: Callable[[BaseException], dict[str, object]],
    ) -> None:
        with self._lock:
            if job.state != QUEUED:
                return
            job.state = RUNNING
            job.started = time.time()
        try:
            result = work()
        except BaseException as exc:  # a model's own code may raise anything, sys.exit included
            error: dict[str, object]
            try:
                error = describe_error(exc)
            except BaseException as failure:  # describe_error itself writes what it can
                error = {
                    "type": exception_name(exc),
                    "message": f"{exception_text(exc)} ({exception_name(failure)})",
                }
            with self._lock:
                job.state, job.error, job.finished = FAILED, error, time.time()
        else:
            with self._lock:
                job.state, job.result, job.finished = DONE, result, time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started; whether it was cancelled."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != QUEUED:
                return False
            job.state = CANCELLED
            job.finished = time.time()
        if job.on_cancel is not None:
            job.on_cancel()
        return True

    def running(self) -> Job | None:
        """The job that is running, if one is."""
        with self._lock:
            return next((j for j in self._jobs.values() if j.state == RUNNING), None)

    def recent(self, limit: int = 20) -> list[Job]:
        """The latest jobs, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.submitted, reverse=True)[:limit]

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Block until a job has finished (for callers that are not a page, such as the tests)."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.future is not None:
            job.future.result(timeout)
        return job

    def close(self) -> None:
        """Stop taking jobs, cancel the queued ones and let a running one finish on its own."""
        cancelled: list[Job] = []
        with self._lock:
            for job in self._jobs.values():
                if job.state == QUEUED:
                    job.state = CANCELLED
                    job.finished = time.time()
                    cancelled.append(job)
        for job in cancelled:
            if job.on_cancel is not None:
                job.on_cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _forget_old(self) -> None:
        """Forget the oldest finished jobs beyond ``keep``, so a long session does not hold every
        result it ever made (a build's result is a key, not the build)."""
        finished = [j for j in self._jobs.values() if not j.active]
        for job in sorted(finished, key=lambda j: j.submitted)[
            : max(0, len(self._jobs) - self._keep)
        ]:
            del self._jobs[job.id]
