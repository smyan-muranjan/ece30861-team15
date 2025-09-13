import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

from src.metrics.local_metrics import LocalMetricsCalculator

# --- Logging setup ---
LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", "0")
LOG_FILE = os.environ.get("LOG_FILE")

log_level_map = {
    "0": logging.WARNING,
    "1": logging.INFO,
    "2": logging.DEBUG,
}
log_level = log_level_map.get(LOG_LEVEL_STR, logging.WARNING)

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
    """
    Reads a file and returns a list of URLs, stripping whitespace.
    """
    logging.info(f"Reading URLs from: {file_path}")
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        logging.info(f"Found {len(urls)} URLs.")
        return urls
    except FileNotFoundError:
        error_msg = f"Error: URL file not found at '{file_path}'."
        logging.error(error_msg)
        print(error_msg + " Please check the path.", file=sys.stderr)
        sys.exit(1)


def create_test_file():
    """Create a clean test file with proper UTF-8 encoding."""
    # Remove old file if it exists
    if os.path.exists('test_urls.txt'):
        try:
            os.remove('test_urls.txt')
        except OSError:
            pass  # Ignore errors if file is locked

    test_urls = [
        "https://github.com/octocat/Hello-World",
        "https://github.com/microsoft/vscode",
        "https://github.com/torvalds/linux"
    ]

    with open('test_urls.txt', 'w', encoding='utf-8') as f:
        for url in test_urls:
            f.write(url + '\n')

    print("Created test_urls.txt with clean UTF-8 encoding")
    return 'test_urls.txt'


def calculate_net_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate the net score based on the weighted formula.
    """
    weights = {
        'license': 0.3,
        'ramp_up_time': 0.2,
        'dataset_and_code_score': 0.2,
        'bus_factor': 0.1,
        'performance_claims': 0.1,
        'dataset_quality': 0.05,
        'code_quality': 0.05,
    }

    net_score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            net_score += metrics[metric] * weight

    return min(1.0, max(0.0, net_score))


def process_urls(urls: List[str]) -> None:
    """
    Processes each URL and prints its analysis in NDJSON format.
    """
    logging.info(f"Processing {len(urls)} URLs.")

    # Initialize the local metrics calculator
    github_token = os.environ.get("GITHUB_TOKEN")
    calculator = LocalMetricsCalculator(github_token)

    try:
        for url in urls:
            start_time = time.time()

            # Analyze the repository using local metrics
            local_metrics = calculator.analyze_repository(url)

            # Calculate net score
            net_score = calculate_net_score(local_metrics)
            net_score_latency = int((time.time() - start_time) * 1000)

            # Create the scorecard
            scorecard: Dict[str, Any] = {
                "name": url.split("/")[-1],
                "category": "MODEL",
                "url": url,
                "net_score": net_score,
                "net_score_latency": net_score_latency,
                "ramp_up_time": local_metrics.get('ramp_up_time', 0.0),
                "ramp_up_time_latency": local_metrics.get(
                    'ramp_up_time_latency', 0),
                "bus_factor": local_metrics.get('bus_factor', 0.0),
                "bus_factor_latency": local_metrics.get(
                    'bus_factor_latency', 0),
                "performance_claims": 0.0,  # Placeholder
                "performance_claims_latency": 0,
                "license": 0.0,  # Placeholder
                "license_latency": 0,
                "size_score": local_metrics.get('size_score', {
                    'raspberry_pi': 0.0,
                    'jetson_nano': 0.0,
                    'desktop_pc': 0.0,
                    'aws_server': 0.0
                }),
                "size_score_latency": local_metrics.get(
                    'size_score_latency', 0),
                "dataset_and_code_score": 0.0,  # Placeholder
                "dataset_and_code_score_latency": 0,
                "dataset_quality": 0.0,  # Placeholder
                "dataset_quality_latency": 0,
                "code_quality": local_metrics.get('code_quality', 0.0),
                "code_quality_latency": local_metrics.get(
                    'code_quality_latency', 0),
            }

            print(json.dumps(scorecard))

    finally:
        # Clean up any temporary directories
        calculator.git_client.cleanup()


def main():
    """Main entry point of the application."""
    if len(sys.argv) == 1:
        # No arguments - create test file and run
        print("No URL file provided. Creating test file...")
        url_file = create_test_file()
        urls = parse_url_file(url_file)
        process_urls(urls)
    elif len(sys.argv) == 2:
        # One argument - use provided file
        url_file = sys.argv[1]
        urls = parse_url_file(url_file)
        process_urls(urls)
    else:
        print("Usage: python -m src.main [URL_FILE]", file=sys.stderr)
        print("  or:  python -m src.main  (to create and use test file)",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
