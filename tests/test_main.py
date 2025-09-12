import os
import json
from src import main


def test_main_runs():
    # This test is a bit obsolete now, but we keep it as a basic sanity check
    # We will expand our E2E tests later.
    pass


def test_parse_url_file_success():
    """Tests that the URL parser correctly reads a valid file."""
    # Create a dummy URL file
    file_path = "test_urls.txt"
    urls_to_write = ["https://github.com/test/repo1",
                     "https://huggingface.co/model2"]
    with open(file_path, "w") as f:
        for url in urls_to_write:
            f.write(url + "\n")

    parsed_urls = main.parse_url_file(file_path)
    assert parsed_urls == urls_to_write

    # Clean up the dummy file
    os.remove(file_path)


def test_process_urls_placeholder():
    """
    Tests the placeholder process_urls function to ensure it produces
    valid NDJSON output for each URL.
    """
    # This test captures stdout to check the output format
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    urls = ["https://github.com/test/repo1",
            "https://huggingface.co/model2"]
    main.process_urls(urls)

    sys.stdout = old_stdout  # Reset stdout

    output = captured_output.getvalue().strip().split('\n')
    assert len(output) == len(urls)

    # Check if each line is valid JSON
    for line in output:
        try:
            data = json.loads(line)
            # Check for a few key fields to ensure format is correct
            assert "net_score" in data
            assert "ramp_up_time" in data
            assert "license" in data
            assert isinstance(data["size_score"], dict)
        except json.JSONDecodeError:
            assert False, "Output is not valid NDJSON"
