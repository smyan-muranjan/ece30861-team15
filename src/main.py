import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

from src.metrics.local_metrics import LocalMetricsCalculator

# --- Logging setup ---
# Adheres to the LOG_FILE and LOG_LEVEL environment variable
# requirements [cite: 424]
LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", "0")
LOG_FILE = os.environ.get("LOG_FILE")

log_level_map = {
    "0": logging.WARNING, "1": logging.INFO, "2": logging.DEBUG
}
log_level = log_level_map.get(LOG_LEVEL_STR, logging.WARNING)

# Configure logging to file or stdout based on environment
if LOG_FILE:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=LOG_FILE,
    )
else:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
# --- End of logging setup ---


def parse_url_file(file_path: str) -> List[str]:
    """Reads a file and returns a list of URLs, stripping whitespace."""
    logging.info(f"Reading URLs from: {file_path}")
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        logging.info(f"Found {len(urls)} URLs.")
        return urls
    except FileNotFoundError:
        # Prints a user-friendly error message and exits as
        # required [cite: 422]
        error_msg = f"Error: URL file not found at '{file_path}'."
        logging.error(error_msg)
        print(error_msg + " Please check the path.", file=sys.stderr)
        sys.exit(1)


def calculate_net_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate the net score using the weighted formula from the project plan.
    """
    # Weights are taken directly from the project plan
    # to match Sarah's priorities
    weights = {
        'license': 0.30,
        'ramp_up_time': 0.20,
        'dataset_and_code_score': 0.15,
        'performance_claims': 0.10,
        'bus_factor': 0.15,  # Adjusted based on re-reading priorities
        'code_quality': 0.05,
        'dataset_quality': 0.05,
    }

    net_score = sum(
        metrics.get(metric, 0.0) * weight
        for metric, weight in weights.items()
    )
    # The score must be in the range [0, 1] [cite: 408]
    return min(1.0, max(0.0, net_score))


async def analyze_url(
    url: str, process_pool: ProcessPoolExecutor
) -> Dict[str, Any]:
    """
    Analyzes a single URL, orchestrates metric calculation, and
    returns the final scorecard.
    """
    start_time = time.time()
    calculator = LocalMetricsCalculator(process_pool)
    local_metrics = await calculator.analyze_repository(url)

    net_score = calculate_net_score(local_metrics)
    total_latency_ms = int((time.time() - start_time) * 1000)

    # The output format strictly follows Table 1 in
    # the project specification [cite: 407, 435]
    scorecard: Dict[str, Any] = {
        "name": url.split("/")[-1],
        "category": "MODEL",
        "url": url,
        "net_score": round(net_score, 2),
        "net_score_latency": total_latency_ms,
        "ramp_up_time": local_metrics.get('ramp_up_time', 0.0),
        "ramp_up_time_latency": local_metrics.get('ramp_up_time_latency', 0),
        "bus_factor": local_metrics.get('bus_factor', 0.0),
        "bus_factor_latency": local_metrics.get('bus_factor_latency', 0),
        "performance_claims": 0.0,  # Placeholder until LLM part is built
        "performance_claims_latency": 0,
        "license": local_metrics.get('license', 0.0),
        "license_latency": local_metrics.get('license_latency', 0),
        "size_score": local_metrics.get('size_score', {}),
        "size_score_latency": local_metrics.get('size_score_latency', 0),
        "dataset_and_code_score": 0.0,  # Placeholder
        "dataset_and_code_score_latency": 0,
        "dataset_quality": 0.0,  # Placeholder
        "dataset_quality_latency": 0,
        "code_quality": local_metrics.get('code_quality', 0.0),
        "code_quality_latency": local_metrics.get('code_quality_latency', 0),
    }
    return scorecard


async def process_urls(urls: List[str]) -> None:
    """
    Processes each URL concurrently using an advanced hybrid model.
    """
    logging.info(f"Processing {len(urls)} URLs with advanced concurrency.")
    # Manages workers based on available CPU cores,
    # as requested by Sarah [cite: 386]
    max_workers = os.cpu_count() or 4
    logging.info(f"Using {max_workers} worker processes.")

    with ProcessPoolExecutor(max_workers=max_workers) as process_pool:
        tasks = [analyze_url(url, process_pool) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logging.error(f"An analysis task failed: {result}")
            else:
                # Prints output to stdout in NDJSON format [cite: 407]
                print(json.dumps(result))


def main():
    """Main entry point of the application."""
    # Handles the `./run URL_FILE` invocation [cite: 399]
    if len(sys.argv) != 2:
        print("Usage: python -m src.main <URL_FILE>", file=sys.stderr)
        sys.exit(1)

    url_file = sys.argv[1]
    urls = parse_url_file(url_file)
    if urls:
        asyncio.run(process_urls(urls))


if __name__ == "__main__":
    main()
