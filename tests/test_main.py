import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src import main

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


@pytest.mark.asyncio
async def test_process_urls():
    """
    Tests the async process_urls function to ensure it produces
    valid NDJSON output for each URL.
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    urls = ["https://github.com/test/repo1", "https://huggingface.co/model2"]

    # Mock the async analyze_url function
    with patch('src.main.analyze_url') as mock_analyze_url:
        # Set up the mock to return different values for each call
        mock_analyze_url.side_effect = [
            {"url": urls[0], "net_score": 0.5},
            {"url": urls[1], "net_score": 0.8},
        ]

        # Await the async function
        await main.process_urls(urls)

    sys.stdout = old_stdout

    output = captured_output.getvalue().strip().split('\n')
    assert len(output) == len(urls)

    # Verify that the mock was called for each URL
    assert mock_analyze_url.call_count == len(urls)

    # Verify the output is valid NDJSON
    for line in output:
        try:
            json.loads(line)
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


@pytest.mark.asyncio
@patch('src.metrics.local_metrics.LocalMetricsCalculator.analyze_repository')
async def test_analyze_url(mock_analyze_repository):
    """Tests the async analyze_url function."""
    # Mock the return value of the underlying analysis
    mock_analyze_repository.return_value = {
        'bus_factor': 0.8,
        'code_quality': 0.9,
        'ramp_up_time': 0.7,
        'license': 1.0,
    }
    url = "https://github.com/test/repo"

    with ProcessPoolExecutor() as pool:
        # Await the async function and pass the required process pool
        scorecard = await main.analyze_url(url, pool)

    assert scorecard['url'] == url
    assert 'net_score' in scorecard
    assert scorecard['net_score'] > 0
    mock_analyze_repository.assert_awaited_once()


def test_main_function_incorrect_args(monkeypatch):
    """
    Tests that the main function exits when called with incorrect arguments.
    """
    # Test with too many arguments (should exit with code 1)
    monkeypatch.setattr(sys, 'argv', ['src/main.py', 'file1.txt', 'file2.txt'])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 1


@patch('src.main.parse_url_file')
@patch('src.main.process_urls')
def test_main_function_with_file(mock_process_urls, mock_parse_url):
    """Tests the main function with a file argument."""
    mock_parse_url.return_value = ['url1', 'url2']
    # Mock the async function as a MagicMock
    mock_process_urls = MagicMock()

    with patch('src.main.process_urls', mock_process_urls):
        with patch('asyncio.run') as mock_asyncio_run:
            with patch.object(sys, 'argv', ['src/main.py', 'some_file.txt']):
                main.main()

            mock_parse_url.assert_called_with('some_file.txt')
            # Check that asyncio.run was called with our mocked function
            mock_asyncio_run.assert_called_once()
