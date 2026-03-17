"""In-process async job manager for extraction/prediction workflows."""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="terra-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def submit_job(
        self,
        job_type: str,
        phases: list[str],
        fn: Callable[..., Any],
        **kwargs,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "type": job_type,
            "status": "queued",
            "created_at": self._now(),
            "updated_at": self._now(),
            "phases": {phase: "queued" for phase in phases},
            "error": None,
            "result": None,
            "logs": [],
        }
        with self._lock:
            self._jobs[job_id] = job

        def runner() -> None:
            self._set_status(job_id, "running")
            try:
                result = fn(job_id=job_id, phase_callback=self.update_phase, **kwargs)
                self._set_result(job_id, result)
                self._set_status(job_id, "completed")
            except Exception as exc:
                self._set_error(job_id, f"{exc}\n{traceback.format_exc()}")
                self._set_status(job_id, "failed")

        self._executor.submit(runner)
        return job_id

    def _set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = status
            job["updated_at"] = self._now()

    def _set_result(self, job_id: str, result: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["result"] = result
            job["updated_at"] = self._now()

    def _set_error(self, job_id: str, error_text: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["error"] = error_text
            job["updated_at"] = self._now()

    def update_phase(self, job_id: str, phase: str, state: str, message: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if phase not in job["phases"]:
                job["phases"][phase] = state
            else:
                job["phases"][phase] = state

            if message:
                job["logs"].append(
                    {
                        "timestamp": self._now(),
                        "phase": phase,
                        "state": state,
                        "message": message,
                    }
                )
            job["updated_at"] = self._now()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job["job_id"],
                "type": job["type"],
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "phases": dict(job["phases"]),
                "error": job["error"],
                "result": job["result"],
                "logs": list(job["logs"]),
            }

    def has_active_job(self, job_type: str) -> bool:
        with self._lock:
            for job in self._jobs.values():
                if job.get("type") == job_type and job.get("status") in {"queued", "running"}:
                    return True
        return False


job_manager = JobManager(max_workers=2)
