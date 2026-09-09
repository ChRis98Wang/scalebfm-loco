from hydra.utils import instantiate
from loguru import logger

from scaleretarget.utils.atomic_io import atomic_joblib_dump

_retargeter = None
_formatter = None


class ProducerFailure:
    """Queue message used to preserve an unexpected producer exception."""

    def __init__(self, exception: Exception):
        self.exception = exception


def ensure_all_motions_accounted_for(
    data_loader,
    *,
    success_num: int,
    processing_failures: int,
) -> None:
    """Accept cached skips while rejecting loader and processing failures."""
    skipped_num = int(getattr(data_loader, "skipped_num", 0))
    loader_failures = int(getattr(data_loader, "failed_num", 0))
    total_num = int(data_loader.data_num)
    completed_num = success_num + skipped_num
    if processing_failures or loader_failures or completed_num != total_num:
        details = getattr(data_loader, "failure_messages", [])
        detail_text = f"; first loader error: {details[0]}" if details else ""
        raise RuntimeError(
            "Retargeting did not complete every motion: "
            f"processed={success_num}, skipped={skipped_num}, "
            f"loader_failed={loader_failures}, worker_failed={processing_failures}, "
            f"total={total_num}{detail_text}"
        )

def initialize_worker(retargeter_config, formatter_config):
    global _retargeter, _formatter
    _retargeter = instantiate(retargeter_config)
    _formatter = instantiate(formatter_config)


def process_single_item(args):
    save_path, frames, extras = args
    if _retargeter is None or _formatter is None:
        raise RuntimeError("Worker models were not initialized")
            
    _retargeter.update(extras)
    qpos_list = _retargeter.retarget(frames)
    formatted_results = _formatter.format(qpos_list, extras)
            
    atomic_joblib_dump(formatted_results, save_path)
    logger.info(f"Retargeted motion saved to {save_path}")

    return save_path


def producer(data_loader, queue):
    try:
        for item in data_loader:
            # The bounded queue blocks here and provides backpressure.
            queue.put(item)
    except Exception as exc:
        queue.put(ProducerFailure(exc))
    finally:
        queue.put(None)
