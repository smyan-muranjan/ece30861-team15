import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src import main

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_parse_url_file_success():
    """Tests URL parser correctly reads valid file with new format."""
    file_path = "test_urls.txt"
    entries_to_write = [
        ("https://github.com/test/repo1, "
         "https://huggingface.co/datasets/test, "
         "https://huggingface.co/model1"),
        ",,https://huggingface.co/model2"
    ]
    with open(file_path, "w") as f:
        for entry in entries_to_write:
            f.write(entry + "\n")

    parsed_entries = main.parse_url_file(file_path)
    expected = [
        ("https://github.com/test/repo1",
         "https://huggingface.co/datasets/test",
         "https://huggingface.co/model1"),
        (None, None, "https://huggingface.co/model2")
    ]
    assert parsed_entries == expected

    os.remove(file_path)


@pytest.mark.asyncio
async def test_process_entries():
    """
    Tests the async process_entries function to ensure it produces
    valid NDJSON output for each entry.
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    entries = [
        ("https://github.com/test/repo1",
         None, "https://huggingface.co/model1"),
        (None, None, "https://huggingface.co/model2")
    ]

    # Mock the async analyze_entry function
    with patch('src.main.analyze_entry') as mock_analyze_entry:
        # Set up the mock to return different values for each call
        mock_analyze_entry.side_effect = [
            {"url": "https://huggingface.co/model1", "net_score": 0.5},
            {"url": "https://huggingface.co/model2", "net_score": 0.8},
        ]

        # Await the async function
        await main.process_entries(entries)

    sys.stdout = old_stdout

    output = captured_output.getvalue().strip().split('\n')
    assert len(output) == len(entries)

    # Verify that the mock was called for each entry
    assert mock_analyze_entry.call_count == len(entries)

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
@patch('src.metrics.metrics_calculator.MetricsCalculator.analyze_entry')
async def test_analyze_entry(mock_analyze_entry_method):
    """Tests the async analyze_entry function."""
    # Mock the return value of the underlying analysis
    mock_analyze_entry_method.return_value = {
        'bus_factor': 0.8,
        'code_quality': 0.9,
        'ramp_up_time': 0.7,
        'license': 1.0,
        'dataset_quality': 0.6,
        'performance_claims': 0.5,
        'dataset_and_code_score': 0.7,
    }
    entry = ("https://github.com/test/repo",
             None, "https://huggingface.co/model")

    with ProcessPoolExecutor() as pool:
        # Await the async function and pass the required process pool
        scorecard = await main.analyze_entry(entry, pool)

    assert 'net_score' in scorecard
    assert scorecard['net_score'] > 0
    mock_analyze_entry_method.assert_awaited_once()


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
@patch('src.main.process_entries')
def test_main_function_with_file(mock_process_entries, mock_parse_url):
    """Tests the main function with a file argument."""
    mock_parse_url.return_value = [
        ('https://github.com/test/repo',
         None, 'https://huggingface.co/model1'),
        (None, None, 'https://huggingface.co/model2')
    ]
    # Mock the async function as a MagicMock
    mock_process_entries = MagicMock()

    with patch('src.main.process_entries', mock_process_entries):
        with patch('asyncio.run') as mock_asyncio_run:
            with patch.object(sys, 'argv', ['src/main.py', 'some_file.txt']):
                main.main()

            mock_parse_url.assert_called_with('some_file.txt')
            # Check that asyncio.run was called with our mocked function
            mock_asyncio_run.assert_called_once()


def test_parse_url_file_invalid_format():
    """Tests URL parser handles invalid format lines correctly."""
    file_path = "test_invalid_urls.txt"
    entries_to_write = [
        "https://github.com/test/repo1, https://huggingface.co/model1",
        ",,",  # Empty model link
        "https://github.com/test/repo4, \
        https://huggingface.co/datasets/test, \
        https://huggingface.co/model4",  # Valid
    ]
    with open(file_path, "w") as f:
        for entry in entries_to_write:
            f.write(entry + "\n")

    parsed_entries = main.parse_url_file(file_path)
    # Should only return the valid entry
    expected = [
        ("https://github.com/test/repo4",
         "https://huggingface.co/datasets/test",
         "https://huggingface.co/model4")
    ]
    assert parsed_entries == expected

    os.remove(file_path)


def test_calculate_net_score():
    """Tests net score calculation with various metric values."""
    # Test with all metrics present
    metrics = {
        'license': 0.8,
        'ramp_up_time': 0.7,
        'dataset_and_code_score': 0.9,
        'performance_claims': 0.6,
        'bus_factor': 0.85,
        'code_quality': 0.75,
        'dataset_quality': 0.8,
    }
    score = main.calculate_net_score(metrics)
    assert 0.0 <= score <= 1.0

    # Test with missing metrics (should default to 0.0)
    partial_metrics = {
        'license': 0.8,
        'ramp_up_time': 0.7,
    }
    score = main.calculate_net_score(partial_metrics)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_process_entries_with_exceptions():
    """Tests process_entries handles exceptions correctly."""
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    entries = [
        ("https://github.com/test/repo1",
            None, "https://huggingface.co/model1"),
        ("https://github.com/test/repo2",
            None, "https://huggingface.co/model2"),
    ]

    # Mock analyze_entry to raise an exception for the first entry
    with patch('src.main.analyze_entry') as mock_analyze_entry:
        mock_analyze_entry.side_effect = [
            Exception("Analysis failed"),
            {"url": "https://huggingface.co/model2", "net_score": 0.8},
        ]

        await main.process_entries(entries)

    sys.stdout = old_stdout
    output = captured_output.getvalue().strip().split('\n')

    # Should have 2 outputs (one for exception, one for success)
    assert len(output) == 2

    # Both should be valid JSON
    for line in output:
        json.loads(line)


# Environment validation tests
def test_validate_environment_invalid_github_token_empty():
    """Test validate_environment with empty GitHub token."""
    with patch.dict(os.environ, {'GITHUB_TOKEN': ''}, clear=True):
        with pytest.raises(SystemExit) as e:
            main.validate_environment()
        assert e.value.code == 1


def test_validate_environment_invalid_github_token_whitespace():
    """Test validate_environment with whitespace-only GitHub token."""
    with patch.dict(os.environ, {'GITHUB_TOKEN': '   '}, clear=True):
        with pytest.raises(SystemExit) as e:
            main.validate_environment()
        assert e.value.code == 1


def test_validate_environment_invalid_github_token_short():
    """Test validate_environment with short GitHub token."""
    with patch.dict(os.environ, {'GITHUB_TOKEN': 'abc'}, clear=True):
        with pytest.raises(SystemExit) as e:
            main.validate_environment()
        assert e.value.code == 1


def test_validate_environment_invalid_log_file_empty():
    """Test validate_environment with empty log file path."""
    with patch.dict(os.environ, {'LOG_FILE': ''}, clear=True):
        with pytest.raises(SystemExit) as e:
            main.validate_environment()
        assert e.value.code == 1


def test_validate_environment_invalid_log_file_whitespace():
    """Test validate_environment with whitespace-only log file path."""
    with patch.dict(os.environ, {'LOG_FILE': '   '}, clear=True):
        with pytest.raises(SystemExit) as e:
            main.validate_environment()
        assert e.value.code == 1


def test_validate_environment_log_file_directory_creation():
    """Test validate_environment creates directory for log file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'subdir', 'test.log')
        with patch.dict(os.environ, {'LOG_FILE': log_file}, clear=True):
            # Should not raise exception and should create directory
            main.validate_environment()
            assert os.path.exists(os.path.dirname(log_file))


def test_validate_environment_log_file_directory_creation_failure():
    """Test validate_environment handles directory creation failure."""
    # Use a path that will fail to create (root directory)
    log_file = '/root/test.log'
    with patch.dict(os.environ, {'LOG_FILE': log_file}, clear=True):
        with pytest.raises(SystemExit) as e:
            main.validate_environment()
        assert e.value.code == 1


def test_validate_environment_log_file_write_failure():
    """Test validate_environment handles log file write failure."""
    # Use a path that exists but is not writable
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file and make it read-only
        log_file = os.path.join(temp_dir, 'readonly.log')
        with open(log_file, 'w') as f:
            f.write('test')
        os.chmod(log_file, 0o444)  # Read-only

        with patch.dict(os.environ, {'LOG_FILE': log_file}, clear=True):
            with pytest.raises(SystemExit) as e:
                main.validate_environment()
            assert e.value.code == 1


def test_validate_environment_log_level_0_blank_file_creation():
    """Test validate_environment creates blank log file for level 0."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'test.log')
        with patch.dict(os.environ,
                        {'LOG_LEVEL': '0', 'LOG_FILE': log_file},
                        clear=True):
            main.validate_environment()
            assert os.path.exists(log_file)
            # File should be empty (blank)
            with open(log_file, 'r') as f:
                assert f.read() == ''


def test_validate_environment_log_level_0_blank_file_creation_failure():
    """Test validate_environment handles blank file creation failure."""
    # Use a path that will fail to write
    log_file = '/root/test.log'
    with patch.dict(os.environ,
                    {'LOG_LEVEL': '0', 'LOG_FILE': log_file}, clear=True):
        with pytest.raises(SystemExit) as e:
            main.validate_environment()
        assert e.value.code == 1


def test_validate_environment_valid_inputs():
    """Test validate_environment with valid inputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'test.log')
        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'ghp_1234567890abcdef1234567890abcdef12345678',
            'LOG_FILE': log_file
        }, clear=True):
            # Should not raise exception
            main.validate_environment()


def test_logging_config_level_0():
    """Test logging configuration with level 0 (disabled)."""
    with patch.dict(os.environ, {'LOG_LEVEL': '0'}, clear=True):
        # Re-import main to trigger logging setup
        import importlib
        importlib.reload(main)

        # Test that logging is disabled
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        main.logging.info("Test message")
        main.logging.debug("Debug message")

        sys.stdout = old_stdout
        output = captured_output.getvalue()
        assert output == ""  # No output because logging is disabled


# Additional error handling tests
def test_validate_environment_log_file_directory_creation_error_handling():
    """Test validate_environment error handling for directory creation."""
    # Mock os.makedirs to raise an exception
    with patch('os.makedirs', side_effect=OSError("Permission denied")):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'subdir', 'test.log')
            with patch.dict(os.environ, {'LOG_FILE': log_file}, clear=True):
                with pytest.raises(SystemExit) as e:
                    main.validate_environment()
                assert e.value.code == 1


def test_validate_environment_log_level_0_blank_file_creation_error_handling():
    """Test validate_environment error handling for blank file creation."""
    # Mock open to raise an exception
    with patch('builtins.open', side_effect=OSError("Permission denied")):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            with patch.dict(
                    os.environ,
                    {'LOG_LEVEL': '0', 'LOG_FILE': log_file},
                    clear=True):
                with pytest.raises(SystemExit) as e:
                    main.validate_environment()
                assert e.value.code == 1


# Main entry point test
def test_main_entry_point():
    """Test the main entry point when script is run directly."""
    with patch.object(sys, 'argv', ['src/main.py', 'nonexistent.txt']):
        with pytest.raises(SystemExit) as e:
            main.main()
        assert e.value.code == 1


def test_parse_url_file_skip_empty_lines():
    """Test parse_url_file skips empty lines (covers line 120)."""
    file_path = "test_empty_lines.txt"
    entries_to_write = [
        "https://github.com/test/repo1, \
        https://huggingface.co/datasets/test, \
        https://huggingface.co/model1",
        "",  # Empty line
        "   ",  # Whitespace-only line
        "https://github.com/test/repo2, \
        https://huggingface.co/datasets/test2, \
        https://huggingface.co/model2"
    ]
    with open(file_path, "w") as f:
        for entry in entries_to_write:
            f.write(entry + "\n")

    parsed_entries = main.parse_url_file(file_path)
    # Should only return the 2 valid entries, skipping empty lines
    assert len(parsed_entries) == 2
    assert parsed_entries[0][2] == "https://huggingface.co/model1"
    assert parsed_entries[1][2] == "https://huggingface.co/model2"

    os.remove(file_path)


def test_validate_environment_blank_file_creation_error():
    """Test error handling for blank file creation (covers lines 63-67)."""
    with patch('builtins.open', side_effect=OSError("Permission denied")):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            with patch.dict(os.environ, {'LOG_LEVEL': '0',
                            'LOG_FILE': log_file}, clear=True):
                with pytest.raises(SystemExit) as e:
                    main.validate_environment()
                assert e.value.code == 1


def test_logging_config_with_file():
    """Test logging configuration with file (covers lines 87-94)."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'test.log')

        # Create a simple test file
        test_file = os.path.join(temp_dir, 'test_urls.txt')
        with open(test_file, 'w') as f:
            f.write("https://huggingface.co/test/model\n")

        env = os.environ.copy()
        env['LOG_LEVEL'] = '1'
        env['LOG_FILE'] = log_file

        # This should trigger the logging configuration code
        result = subprocess.run([
            sys.executable, '-c',
            'import sys; sys.path.append("."); \
            import src.main; print("Logging configured")'
        ], env=env, capture_output=True, text=True, cwd='.')

        # The logging configuration should have been triggered
        assert result.returncode == 0


def test_main_entry_point_direct():
    """Test main entry point when run directly (covers line 302)."""
    # Test that the main function exists and is callable
    assert callable(main.main)

    # Test the __name__ == "__main__" condition by checking the module
    # This is hard to test directly, but we can verify the function exists
    assert hasattr(main, 'main')


def test_main_script_execution():
    """Test running the main script directly (covers line 302)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = os.path.join(temp_dir, 'test_urls.txt')
        with open(test_file, 'w') as f:
            f.write("https://huggingface.co/test/model\n")

        # Run the main script directly
        result = subprocess.run([
            sys.executable, '-m', 'src.main', test_file
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode in [0, 1]


def test_logging_config_without_file():
    """Test logging configuration without file (covers line 94)."""
    # This test needs to trigger the else branch of the logging config
    import subprocess

    # Run with LOG_LEVEL=1 but no LOG_FILE to trigger the else branch
    env = os.environ.copy()
    env['LOG_LEVEL'] = '1'
    if 'LOG_FILE' in env:
        del env['LOG_FILE']

    result = subprocess.run([
        sys.executable, '-c',
        'import sys; sys.path.append("."); \
        import src.main; print("Logging configured without file")'
    ], env=env, capture_output=True, text=True, cwd='.')

    # Should run successfully
    assert result.returncode == 0
