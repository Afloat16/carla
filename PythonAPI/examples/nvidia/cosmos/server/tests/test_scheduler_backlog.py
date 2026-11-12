"""Queued work must reserve shared GPUs before the dispatch loop yields."""

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from carla_cosmos.client import sha256_file
from carla_cosmos.contracts import BUILTIN_CONTRACTS, JobRequest
from carla_cosmos.synthetic import make_clip
from carla_cosmos_server.config import Settings
from carla_cosmos_server.scheduler import Scheduler
from carla_cosmos_server.store import JobRecord, Store
from carla_cosmos_server.workers_rpc import ProgressCb, RunOutcome, WorkerHandle


def _queue_clip(store: Store, path: Path, name: str, backend: str, frames: int) -> JobRecord:
    clip = make_clip(path, frames=frames, kinds=("depth",), clip_id=name)
    view = clip.manifest.camera_names[0]
    video = clip.video("depth", view)
    blob_id = sha256_file(video)
    destination = store.blob_path(blob_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(video, destination)
    store.add_blob(blob_id, destination.stat().st_size, "video/mp4", video.name)
    request = JobRequest(backend=backend, prompt="test", controls={"depth": {"blob": blob_id}})
    return store.create_job("test", request, clip.manifest.model_dump(), [view], [blob_id])


def test_queued_jobs_reserve_shared_gpus_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real queue/preparation; replace only the model worker transport."""
    settings = Settings(state_dir=tmp_path / "state", run_dir=tmp_path / "run")
    settings.ensure_dirs()
    store = Store(settings.db_file, settings.blobs_dir, settings.jobs_dir)
    workers = []
    jobs = []
    for name, backend, gpus, frames in (("nano", "cosmos3-nano", [0, 1], 16),
                                       ("super", "cosmos3-super", [1], 16),
                                       ("transfer", "transfer2.5", [2], 93)):
        jobs.append(_queue_clip(store, tmp_path / "clips", name, backend, frames))
        workers.append(WorkerHandle(name=name, backends=[backend], gpus=gpus,
                                    socket=settings.run_dir / f"{name}.sock", state="ready", smoke_ok=True))

    async def run() -> None:
        scheduler = Scheduler(store, workers, dict(BUILTIN_CONTRACTS))
        entered = {worker.name: asyncio.Event() for worker in workers}
        release = {worker.name: asyncio.Event() for worker in workers}

        async def run_worker(worker: WorkerHandle, payload: dict[str, Any], _progress: ProgressCb) -> RunOutcome:
            assert payload["job_id"] == worker.current_job
            assert Path(payload["inputs"]["controls"]["depth"]["path"]).is_file()
            entered[worker.name].set()
            await release[worker.name].wait()
            worker.current_job = None
            return RunOutcome(ok=True)

        monkeypatch.setattr(WorkerHandle, "run", run_worker)
        try:
            await scheduler.start()
            await asyncio.wait_for(asyncio.gather(entered["nano"].wait(), entered["transfer"].wait()), timeout=5)
            assert store.get_job(jobs[1].id).status == "queued"
            assert not entered["super"].is_set()
            release["nano"].set()
            release["transfer"].set()
            await asyncio.wait_for(entered["super"].wait(), timeout=5)
            assert store.get_job(jobs[0].id).status == "done"
            release["super"].set()

            async def completed() -> None:
                while any(store.get_job(job.id).status != "done" for job in jobs):
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(completed(), timeout=5)
            assert all(worker.current_job is None for worker in workers)
        finally:
            for event in release.values():
                event.set()
            await scheduler.stop()
            store.close()

    asyncio.run(run())
