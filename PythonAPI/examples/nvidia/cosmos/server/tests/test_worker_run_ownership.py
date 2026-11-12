"""A disconnected request cannot release a model that is still executing."""

import argparse
import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import pytest

from cosmos_workers.common.base import CancelledJob, RunContext, RunResult, Worker, _Server


class GatedWorker(Worker):
    """Use a real inference thread whose completion the test controls."""

    def __init__(self) -> None:
        super().__init__(argparse.Namespace())
        self.state = "ready"
        self.release = threading.Event()
        self.calls: list[str] = []

    def run(self, job: dict[str, Any], ctx: RunContext) -> RunResult:
        self.calls.append(job["job_id"])
        ctx.progress(0.1, "started")
        if job["job_id"] == "first":
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release the model")
        ctx.check_cancelled()
        if job.get("outcome") == "failure":
            raise ValueError("model failed")
        if job.get("outcome") == "cancelled":
            raise CancelledJob(ctx.job_id)
        return RunResult(files=[], manifest={"job": ctx.job_id})


class RecordingExecutor(ThreadPoolExecutor):
    """Keep the real submitted futures so discarded queue work is observable."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.submitted: list[Future] = []

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        future = super().submit(fn, *args, **kwargs)
        self.submitted.append(future)
        return future


@pytest.mark.parametrize("disconnect", ["send-error", "handler-cancel"])
def test_worker_stays_busy_after_disconnected_run(disconnect: str) -> None:
    """A second RPC is refused until the first model thread actually returns."""
    worker = GatedWorker()

    async def run() -> None:
        server = _Server(worker, "unused.sock")
        progress_sent = asyncio.Event()

        async def send(message: dict[str, Any]) -> None:
            if message["event"] == "progress":
                progress_sent.set()
                if disconnect == "send-error":
                    raise ConnectionResetError("request connection closed")

        messages: list[dict[str, Any]] = []

        async def collect(message: dict[str, Any]) -> None:
            messages.append(message)

        handler = asyncio.create_task(server._run(1, {"job_id": "first"}, send))  # pylint: disable=protected-access
        try:
            await asyncio.wait_for(progress_sent.wait(), timeout=5)
            if disconnect == "handler-cancel":
                handler.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await handler
            else:
                with pytest.raises(ConnectionResetError):
                    await handler
            assert worker.state == "busy"
            assert worker.current is not None and worker.current.job_id == "first"
            assert worker.current.cancelled
            await server._run(2, {"job_id": "second"}, collect)  # pylint: disable=protected-access
            assert messages[-1]["event"] == "failed" and "busy" in messages[-1]["error"]
            assert worker.calls == ["first"]
            worker.release.set()

            async def idle() -> None:
                while worker.state != "ready":
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(idle(), timeout=5)
            assert worker.current is None
            await server._run(3, {"job_id": "third"}, collect)  # pylint: disable=protected-access
            assert messages[-1]["event"] == "done"
            assert worker.calls == ["first", "third"]
        finally:
            worker.release.set()
            if not handler.done():
                handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["done", "failure", "cancelled"])
def test_worker_releases_ownership_after_model_completion(outcome: str) -> None:
    """All ordinary terminal outcomes still restore the worker for its next job."""
    worker = GatedWorker()
    messages: list[dict[str, Any]] = []

    async def run() -> None:
        server = _Server(worker, "unused.sock")

        async def collect(message: dict[str, Any]) -> None:
            messages.append(message)

        await server._run(1, {"job_id": "normal", "outcome": outcome}, collect)  # pylint: disable=protected-access

    asyncio.run(run())
    assert worker.state == "ready" and worker.current is None
    assert messages[-1]["event"] == ("done" if outcome == "done" else "failed")
    if outcome == "cancelled":
        assert messages[-1]["cancelled"] is True
    if outcome == "failure":
        assert messages[-1]["error"] == "ValueError: model failed"


@pytest.mark.parametrize("mode", ["rejected", "discarded"])
def test_worker_recovers_when_executor_never_starts_model(mode: str) -> None:
    """Rejected submission or discarded queued work must not leave busy forever."""
    worker = GatedWorker()
    executor = RecordingExecutor()
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def occupy_pool() -> None:
        blocker_started.set()
        if not release_blocker.wait(timeout=5):
            raise TimeoutError("test did not release the executor")

    if mode == "discarded":
        executor.submit(occupy_pool)
        assert blocker_started.wait(timeout=5)
    else:
        executor.shutdown()

    async def run() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        server = _Server(worker, "unused.sock")
        messages: list[dict[str, Any]] = []

        async def collect(message: dict[str, Any]) -> None:
            messages.append(message)

        handler = asyncio.create_task(server._run(1, {"job_id": "never-started"}, collect))  # pylint: disable=protected-access
        if mode == "discarded":
            async def submitted() -> None:
                while len(executor.submitted) < 2:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(submitted(), timeout=5)
            executor.shutdown(wait=False, cancel_futures=True)
            with pytest.raises(asyncio.CancelledError):
                await handler
            assert executor.submitted[1].cancelled()
        else:
            await handler
            assert messages[-1]["event"] == "failed"
        assert not worker.calls
        assert worker.state == "ready" and worker.current is None
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        await server._run(2, {"job_id": "recovered"}, collect)  # pylint: disable=protected-access
        assert messages[-1]["event"] == "done" and worker.calls == ["recovered"]

    try:
        asyncio.run(run())
    finally:
        release_blocker.set()
        executor.shutdown(wait=True)
