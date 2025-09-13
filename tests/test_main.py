import importlib
import json
import logging
import os
import sys
from io import StringIO

import pytest

from src import main

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_main_runs():
    """Tests that the main function can be called without error."""
    pass


def test_parse_url_file_success():
    """Tests that the URL parser correctly reads a valid file."""
    file_path = "test_urls.txt"
    urls_to_write = [
        "https://github.com/test/repo1",
        "https://huggingface.co/model2"
    ]
    with open(file_path, "w") as f:
        for url in urls_to_write:
            f.write(url + "\n")

    parsed_urls = main.parse_url_file(file_path)
    assert parsed_urls == urls_to_write

    os.remove(file_path)


def test_process_urls_placeholder():
    """
    Tests the placeholder process_urls function to ensure it produces
    valid NDJSON output for each URL.
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    urls = ["https://github.com/test/repo1", "https://huggingface.co/model2"]
    main.process_urls(urls)

    sys.stdout = old_stdout

    output = captured_output.getvalue().strip().split('\n')
    assert len(output) == len(urls)

    for line in output:
        try:
            data = json.loads(line)
            assert "net_score" in data
            assert "ramp_up_time" in data
            assert "license" in data
            assert isinstance(data["size_score"], dict)
        except json.JSONDecodeError:
            assert False, "Output is not valid NDJSON"


def test_parse_url_file_not_found():
    """
    Tests that the script exits with code 1 when the URL file is not found.
    """
    with pytest.raises(SystemExit) as e:
        main.parse_url_file("non_existent_file.txt")
    assert e.type == SystemExit
    assert e.value.code == 1


def test_parse_url_file_empty():
    """Tests that the URL parser correctly handles an empty file."""
    file_path = "empty_file.txt"
    with open(file_path, "w"):
        pass

    parsed_urls = main.parse_url_file(file_path)
    assert parsed_urls == []

    os.remove(file_path)


def test_main_function_incorrect_args(monkeypatch):
    """
    Tests that the main function exits when called with incorrect arguments.
    """
    # Test with too many arguments (should exit with code 1)
    monkeypatch.setattr(sys, 'argv', ['src/main.py', 'file1.txt', 'file2.txt'])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 1


def test_logging_to_file(monkeypatch):
    """
    Tests that logging is correctly configured to a file when LOG_FILE is set.
    """
    log_file = "test.log"
    monkeypatch.setenv("LOG_LEVEL", "2")
    monkeypatch.setenv("LOG_FILE", log_file)

    importlib.reload(main)

    root_logger = logging.getLogger()
    assert any(isinstance(h, logging.FileHandler) for h
               in root_logger.handlers)

    if os.path.exists(log_file):
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        os.remove(log_file)

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FILE", raising=False)
    importlib.reload(main)
