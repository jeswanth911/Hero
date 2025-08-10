import asyncio
import logging
import uuid
from typing import Callable, Any, Dict, Optional, Awaitable, List
from fastapi import BackgroundTasks
import time

# Optional: Use Celery if available; fallback to FastAPI BackgroundTasks
try:
    from celery import Celery, states
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Global registry for workflows and statuses (in-memory for demo)
WORKFLOW_REGISTRY: Dict[str, Dict[str, Any]] = {}
PROGRESS_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Setup logger
logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)

# ========== Utilities ==========

def exponential_backoff(attempt: int, base: float = 1.0, factor: float = 2.0, max_delay: float = 60.0) -> float:
    return min(base * (factor ** attempt), max_delay)

def generate_job_id() -> str:
    return str(uuid.uuid4())

# ========== Celery Setup (if available) ==========

if CELERY_AVAILABLE:
    celery_app = Celery("orchestrator", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

# ========== Orchestrator Class ==========

class Orchestrator:
    """
    Asynchronous job orchestrator for multi-step workflows.
    Supports both Celery and FastAPI BackgroundTasks.
    """
    def __init__(self, use_celery: bool = False):
        self.use_celery = use_celery and CELERY_AVAILABLE
        self.logger = logger

    def submit_workflow(
        self,
        workflow_steps: List[Callable[..., Awaitable[Any]]],
        step_args: List[Dict[str, Any]],
        background_tasks: Optional[BackgroundTasks] = None,
        user_id: Optional[str] = None,
        notify_callback: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """
        Submit a multi-step workflow.
        Args:
            workflow_steps: List of async callables for each step.
            step_args: List of dicts with arguments for each step.
            background_tasks: FastAPI BackgroundTasks for async execution (if not using Celery).
            user_id: Optional user id for tracking/alerting.
            notify_callback: Optional function (job_id, status_msg) for alerts.

        Returns:
            job_id: Job identifier for tracking.
        """
        job_id = generate_job_id()
        WORKFLOW_REGISTRY[job_id] = {
            "status": "queued",
            "progress": 0,
            "step": 0,
            "result": None,
            "error": None,
            "user_id": user_id,
            "steps_count": len(workflow_steps)
        }

        if self.use_celery:
            # Submit to Celery
            celery_task = run_workflow_celery.apply_async(
                args=[job_id, workflow_steps, step_args, notify_callback]
            )
            WORKFLOW_REGISTRY[job_id]["celery_task_id"] = celery_task.id
        else:
            # Use FastAPI BackgroundTasks or run in asyncio loop
            if background_tasks:
                background_tasks.add_task(self._run_workflow, job_id, workflow_steps, step_args, notify_callback)
            else:
                asyncio.create_task(self._run_workflow(job_id, workflow_steps, step_args, notify_callback))
        return job_id

    async def _run_workflow(
        self,
        job_id: str,
        workflow_steps: List[Callable[..., Awaitable[Any]]],
        step_args: List[Dict[str, Any]],
        notify_callback: Optional[Callable[[str, str], None]],
        max_retries: int = 3,
    ):
        """
        Run a workflow asynchronously, step by step, with retries and progress tracking.
        """
        result = None
        for idx, (step, args) in enumerate(zip(workflow_steps, step_args)):
            retries = 0
            while retries <= max_retries:
                try:
                    WORKFLOW_REGISTRY[job_id]["status"] = f"step_{idx+1}_running"
                    WORKFLOW_REGISTRY[job_id]["step"] = idx + 1
                    WORKFLOW_REGISTRY[job_id]["progress"] = int((idx / len(workflow_steps)) * 100)
                    self.logger.info(f"Job {job_id} Step {idx+1}: {step.__name__}, args={args}")
                    result = await step(**args)
                    break
                except Exception as e:
                    retries += 1
                    delay = exponential_backoff(retries)
                    self.logger.error(f"Job {job_id} Step {idx+1} failed (attempt {retries}): {e}")
                    if retries > max_retries:
                        WORKFLOW_REGISTRY[job_id]["status"] = "failed"
                        WORKFLOW_REGISTRY[job_id]["error"] = str(e)
                        if notify_callback:
                            notify_callback(job_id, f"Workflow failed at step {idx+1}: {e}")
                        return
                    else:
                        await asyncio.sleep(delay)
            # Update progress
            WORKFLOW_REGISTRY[job_id]["progress"] = int(((idx+1) / len(workflow_steps)) * 100)
            if notify_callback:
                notify_callback(job_id, f"Step {idx+1} completed.")

        WORKFLOW_REGISTRY[job_id]["status"] = "completed"
        WORKFLOW_REGISTRY[job_id]["result"] = result
        WORKFLOW_REGISTRY[job_id]["progress"] = 100
        if notify_callback:
            notify_callback(job_id, "Workflow completed.")

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the current status and progress of a job.
        """
        return WORKFLOW_REGISTRY.get(job_id, {"status": "unknown"})

    def cancel_job(self, job_id: str):
        """
        Attempt to cancel a running job (best-effort, depends on backend).
        """
        if self.use_celery and "celery_task_id" in WORKFLOW_REGISTRY[job_id]:
            celery_app.control.revoke(WORKFLOW_REGISTRY[job_id]["celery_task_id"], terminate=True)
            WORKFLOW_REGISTRY[job_id]["status"] = "cancelled"
        else:
            # For asyncio, best-effort: mark as cancelled
            WORKFLOW_REGISTRY[job_id]["status"] = "cancelled"

# ========== Celery Task (If using Celery) ==========

if CELERY_AVAILABLE:
    @celery_app.task(bind=True)
    def run_workflow_celery(self, job_id, workflow_steps, step_args, notify_callback, max_retries=3):
        result = None
        for idx, (step, args) in enumerate(zip(workflow_steps, step_args)):
            retries = 0
            while retries <= max_retries:
                try:
                    WORKFLOW_REGISTRY[job_id]["status"] = f"step_{idx+1}_running"
                    WORKFLOW_REGISTRY[job_id]["step"] = idx + 1
                    WORKFLOW_REGISTRY[job_id]["progress"] = int((idx / len(workflow_steps)) * 100)
                    logger.info(f"[Celery] Job {job_id} Step {idx+1}: {step.__name__}, args={args}")
                    result = step(**args)
                    break
                except Exception as e:
                    retries += 1
                    delay = exponential_backoff(retries)
                    logger.error(f"[Celery] Job {job_id} Step {idx+1} failed (attempt {retries}): {e}")
                    if retries > max_retries:
                        WORKFLOW_REGISTRY[job_id]["status"] = "failed"
                        WORKFLOW_REGISTRY[job_id]["error"] = str(e)
                        if notify_callback:
                            notify_callback(job_id, f"Workflow failed at step {idx+1}: {e}")
                        return
                    else:
                        time.sleep(delay)
            # Update progress
            WORKFLOW_REGISTRY[job_id]["progress"] = int(((idx+1) / len(workflow_steps)) * 100)
            if notify_callback:
                notify_callback(job_id, f"Step {idx+1} completed.")
        WORKFLOW_REGISTRY[job_id]["status"] = "completed"
        WORKFLOW_REGISTRY[job_id]["result"] = result
        WORKFLOW_REGISTRY[job_id]["progress"] = 100
        if notify_callback:
            notify_callback(job_id, "Workflow completed.")

# ========== Example Hooks ==========

def alert_on_failure(job_id: str, message: str):
    logger.warning(f"[ALERT] Job {job_id}: {message}")

# ========== Example Integration Points ==========

# Usage example:
# orchestrator = Orchestrator(use_celery=False)
# job_id = orchestrator.submit_workflow(
#     [async_ingest, async_clean, async_analyze, async_model, async_export, async_query],
#     [dict(...), dict(...), ...],
#     background_tasks=background_tasks,
#     user_id="user123",
#     notify_callback=alert_on_failure
# )

# In API, expose /status/{job_id} and /cancel/{job_id} endpoints to call get_status/cancel_job.



# api/routes.py
from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from ingestion.ingestion_service import ingest_file  # implement this
from query_engine.nl_to_sql import query_data       # implement this

router = APIRouter(tags=["amma"])

@router.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...), dataset_name: Optional[str] = Form(None)):
    """
    Accepts a single uploaded file and returns dataset_id (or error).
    """
    # Read bytes and pass to ingestion service
    content = await file.read()
    result = await ingest_file(content, filename=file.filename, dataset_name=dataset_name)
    return result

@router.post("/query")
async def query_endpoint(question: str = Form(...), dataset_id: str = Form(...)):
    """
    Accepts NL question + dataset_id, returns answer.
    """
    answer = await query_data(question, dataset_id)
    return answer
    
