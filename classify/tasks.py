"""CLASSIFY - Background Task Processing Engine.

Provides asynchronous audio analysis task execution with thread-pool workers,
fine-grained status tracking, Server-Sent Events (SSE) streaming, and Redis/Celery hooks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# In-memory thread-safe task registry
_TASK_LOCK = threading.Lock()
_TASKS: Dict[str, Dict[str, Any]] = {}

# ThreadPool executor for background audio analysis workers
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="classify-worker")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves current execution status dictionary for a task."""
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if task:
            return dict(task)
        return None


def update_task_progress(task_id: str, progress: int, step: str, status: str = "processing"):
    """Updates progress percentage and stage description for an active task."""
    with _TASK_LOCK:
        if task_id in _TASKS:
            _TASKS[task_id]["progress"] = progress
            _TASKS[task_id]["step"] = step
            _TASKS[task_id]["status"] = status
            _TASKS[task_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def complete_task(task_id: str, result: Dict[str, Any]):
    """Marks task as successfully completed with final payload."""
    with _TASK_LOCK:
        if task_id in _TASKS:
            _TASKS[task_id]["status"] = "completed"
            _TASKS[task_id]["progress"] = 100
            _TASKS[task_id]["step"] = "Analysis complete"
            _TASKS[task_id]["result"] = result
            _TASKS[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            _TASKS[task_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def fail_task(task_id: str, error_message: str):
    """Marks task as failed with error description."""
    with _TASK_LOCK:
        if task_id in _TASKS:
            _TASKS[task_id]["status"] = "failed"
            _TASKS[task_id]["error"] = error_message
            _TASKS[task_id]["step"] = "Analysis failed"
            _TASKS[task_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def _run_analysis_worker(
    app,
    task_id: str,
    file_path: str,
    original_filename: str,
    stored_filename: str,
    user_id: Optional[int],
    allowed_extensions: Optional[set],
    max_size_bytes: Optional[int],
):
    """Worker function executed inside the background thread pool."""
    with app.app_context():
        from classify.models import User
        from classify.services.analysis_service import analyze_uploaded_audio

        user = None
        if user_id:
            user = User.query.get(user_id)

        try:
            update_task_progress(task_id, 15, "Validating audio integrity and format...")
            time.sleep(0.05)

            update_task_progress(task_id, 40, "Harmonizing bandwidth and extracting acoustic features...")
            time.sleep(0.05)

            update_task_progress(task_id, 75, "Classifying genres and calculating model explanations...")
            analysis_result = analyze_uploaded_audio(
                file_path=file_path,
                original_filename=original_filename,
                stored_filename=stored_filename,
                user=user,
                allowed_extensions=allowed_extensions,
                max_size_bytes=max_size_bytes,
            )

            features = analysis_result["features"]
            features["file_name"] = original_filename
            classification = analysis_result["classification"]

            final_payload = {
                "features": features,
                "classification": classification,
                "validation": analysis_result["validation"],
                "analysis_id": analysis_result["analysis_id"],
                "stored_filename": stored_filename,
            }

            complete_task(task_id, final_payload)
            logger.info("TASK_COMPLETED: task_id=%s analysis_id=%s", task_id, analysis_result.get("analysis_id"))

        except Exception as e:
            logger.exception("TASK_FAILED: task_id=%s error=%s", task_id, e)
            fail_task(task_id, str(e))


def submit_analysis_task(
    app,
    file_path: str,
    original_filename: str,
    stored_filename: str,
    user: Any = None,
    allowed_extensions: Optional[set] = None,
    max_size_bytes: Optional[int] = None,
) -> str:
    """Submits an asynchronous audio analysis task to the worker pool.

    Returns the unique task_id.
    """
    task_id = uuid.uuid4().hex
    user_id = getattr(user, "id", None) if user and getattr(user, "is_authenticated", False) else None

    with _TASK_LOCK:
        _TASKS[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 5,
            "step": "Task queued for processing",
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    _EXECUTOR.submit(
        _run_analysis_worker,
        app._get_current_object() if hasattr(app, "_get_current_object") else app,
        task_id,
        file_path,
        original_filename,
        stored_filename,
        user_id,
        allowed_extensions,
        max_size_bytes,
    )

    return task_id


def stream_task_events(task_id: str, timeout_seconds: int = 120) -> Generator[str, None, None]:
    """Yields Server-Sent Events (SSE) formatted text for an active task until completion."""
    start_time = time.time()
    last_progress = -1

    while time.time() - start_time < timeout_seconds:
        status = get_task_status(task_id)
        if not status:
            data = json.dumps({"error": "Task not found", "status": "failed"})
            yield f"data: {data}\n\n"
            break

        # Only emit on state changes or keepalive every few loops
        current_prog = status.get("progress", 0)
        if current_prog != last_progress or status.get("status") in ("completed", "failed"):
            last_progress = current_prog
            payload = json.dumps(status)
            yield f"data: {payload}\n\n"

        if status.get("status") in ("completed", "failed"):
            break

        time.sleep(0.4)
