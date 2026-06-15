import logging
from typing import List, Callable, Any
from joblib import Parallel, delayed

logger = logging.getLogger("beverage_cleaner.utils")

def chunk_data(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """Helper to partition a list into chunks of a given size."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def parallel_process(
    data: List[Any],
    worker_fn: Callable[[List[Any]], List[Any]],
    n_jobs: int = -1,
    chunk_size: int = 1000,
) -> List[Any]:
    """
    Splits data into chunks and processes them in parallel using joblib.
    Excellent for scaling CPU-bound text cleaning pipelines.
    
    Args:
        data: List of items to process.
        worker_fn: Function that accepts a list of items and returns a list of processed items.
        n_jobs: Number of parallel jobs. -1 uses all available CPU cores.
        chunk_size: Number of items per chunk/process.
        
    Returns:
        List of all processed items joined back in order.
    """
    if not data:
        return []

    # Partition data
    chunks = chunk_data(data, chunk_size)
    num_chunks = len(chunks)
    logger.info(
        f"Spinning up parallel execution: processing {len(data)} items "
        f"across {num_chunks} chunks using {n_jobs} jobs (chunk_size={chunk_size})."
    )

    try:
        # Run parallel chunks
        # backend='multiprocessing' or 'loky' (default)
        results = Parallel(n_jobs=n_jobs)(
            delayed(worker_fn)(chunk) for chunk in chunks
        )
        
        # Flatten the list of lists
        flattened = [item for sublist in results for item in sublist]
        return flattened
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}. Falling back to sequential execution.")
        # Fall back to sequential processing if multiprocessing fails
        return worker_fn(data)
