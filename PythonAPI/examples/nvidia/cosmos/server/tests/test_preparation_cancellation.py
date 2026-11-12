"""Accepted preparation cancellation remains terminal when in-flight work ends."""

import asyncio
import hashlib
import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest

from carla_cosmos.contracts import BUILTIN_CONTRACTS, JobRequest
from carla_cosmos.synthetic import make_clip
from carla_cosmos_server.config import Settings
from carla_cosmos_server.scheduler import Scheduler
from carla_cosmos_server.store import JobRecord, Store
from carla_cosmos_server.workers_rpc import ProgressCb, RunOutcome, WorkerHandle


def _queue_scene(store: Store, tmp_path: Path, phase: str) -> JobRecord:
    clip = make_clip(tmp_path / "clips", kinds=(), clip_id="cancellation")
    scene = tmp_path / "scene.zip"
    if phase == "render":
        with zipfile.ZipFile(scene, "w") as archive:
            archive.writestr("scene.json", "{}")
    else:
        scene.write_bytes(b"not a zip archive")
    blob_id = hashlib.sha256(scene.read_bytes()).hexdigest()
    blob_path = store.blob_path(blob_id)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(scene.read_bytes())
    store.add_blob(blob_id, blob_path.stat().st_size, "application/zip", scene.name)
    request = JobRequest(backend="transfer2.5-av", prompt="test", controls={"hdmap_bbox": {"scene": blob_id}})
    return store.create_job("test", request, clip.manifest.model_dump(), clip.manifest.camera_names, [blob_id])


def _assert_outcome(final: JobRecord, accepted: JobRecord | None, phase: str) -> None:
    assert final.status == ("cancelled" if accepted is not None else "failed")
    if accepted is not None:
        assert final.error == accepted.error == "cancelled during preparation"
        assert final.finished == accepted.finished
    elif phase == "shutdown":
        assert final.error == "server shutting down"
    else:
        assert ("not a zip archive" if phase == "prepare" else "renderer failed") in final.error


@pytest.mark.parametrize("phase", ["prepare", "render", "shutdown"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_preparation_completion_preserves_accepted_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, cancelled: bool,
) -> None:
    """Use real SQLite/prepare/scene extraction; gate only the slow work boundary."""
    settings = Settings(state_dir=tmp_path / "state", run_dir=tmp_path / "run")
    settings.ensure_dirs()
    store = Store(settings.db_file, settings.blobs_dir, settings.jobs_dir)
    job = _queue_scene(store, tmp_path, phase)
    model = WorkerHandle(name="model", backends=["transfer2.5-av"], gpus=[0],
                         socket=settings.run_dir / "model.sock", state="ready", smoke_ok=True)
    renderer = WorkerHandle(name="renderer", backends=["wsm-renderer"], gpus=[0],
                            socket=settings.run_dir / "renderer.sock", state="ready", smoke_ok=True)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        scheduler = Scheduler(store, [model, renderer], dict(BUILTIN_CONTRACTS))
        entered = asyncio.Event()
        prepared = asyncio.Event()
        release_thread = threading.Event()
        release_renderer = asyncio.Event()
        original_prepare = scheduler.prepare

        def prepare(record: JobRecord) -> dict[str, Any]:
            loop.call_soon_threadsafe(entered.set)
            try:
                if not release_thread.wait(timeout=5):
                    raise TimeoutError("test did not release preparation")
                return original_prepare(record)
            finally:
                loop.call_soon_threadsafe(prepared.set)

        async def run_worker(worker: WorkerHandle, _payload: dict[str, Any], _progress: ProgressCb) -> RunOutcome:
            assert worker is renderer, "a cancelled or invalid preparation must never reach the model"
            entered.set()
            await release_renderer.wait()
            return RunOutcome(ok=False, error="renderer failed to read scene", traceback="late renderer failure")

        if phase != "render":
            monkeypatch.setattr(scheduler, "prepare", prepare)
        monkeypatch.setattr(WorkerHandle, "run", run_worker)
        try:
            await scheduler.start()
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert store.get_job(job.id).status == "preparing"
            if cancelled:
                assert await scheduler.cancel(job.id)
                accepted = store.get_job(job.id)
                assert accepted.status == "cancelled"
            if phase == "shutdown":
                await scheduler.stop()
            release_thread.set()
            release_renderer.set()
            if phase != "render":
                await asyncio.wait_for(prepared.wait(), timeout=5)

            async def finished() -> None:
                while job.id in scheduler._running:  # pylint: disable=protected-access
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(finished(), timeout=5)
            _assert_outcome(store.get_job(job.id), accepted if cancelled else None, phase)
            assert model.current_job is None
        finally:
            release_thread.set()
            release_renderer.set()
            await scheduler.stop()

    try:
        asyncio.run(run())
    finally:
        store.close()
