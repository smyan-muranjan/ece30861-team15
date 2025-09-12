import sys
import os
import json
import logging
from typing import List, Dict, Any

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
        with open(file_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        logging.info(f"Found {len(urls)} URLs.")
        return urls
    except FileNotFoundError:
        logging.error(f"Error: URL file not found at '{file_path}'.")
        print(f"Error: URL file not found at '{file_path}'. Please check the path.", file=sys.stderr)
        sys.exit(1)


def process_urls(urls: List[str]) -> None:
    """
    Processes each URL and prints its analysis in NDJSON format.
    NOTE: This is a placeholder for Task 1 to meet I/O requirements.
    """
    logging.info(f"Processing {len(urls)} URLs.")
    for url in urls:
        scorecard: Dict[str, Any] = {
            "name": url.split("/")[-1], "category": "MODEL", "url": url, "net_score": 0.75,
            "net_score_latency": 150, "ramp_up_time": 0.8, "ramp_up_time_latency": 20,
            "bus_factor": 0.6, "bus_factor_latency": 30, "performance_claims": 0.9,
            "performance_claims_latency": 15, "license": 1.0, "license_latency": 5,
            "size_score": {"raspberry_pi": 1.0, "jetson_nano": 1.0, "desktop_pc": 1.0, "aws_server": 1.0},
            "size_score_latency": 10, "dataset_and_code_score": 0.7, "dataset_and_code_score_latency": 25,
            "dataset_quality": 0.8, "dataset_quality_latency": 20, "code_quality": 0.6,
            "code_quality_latency": 25,
        }
        print(json.dumps(scorecard))


def main():
    """Main entry point of the application."""
    if len(sys.argv) != 2:
        print("Usage: ./run <URL_FILE>", file=sys.stderr)
        sys.exit(1)

    url_file = sys.argv[1]
    urls = parse_url_file(url_file)
    process_urls(urls)


if __name__ == "__main__":
    main()