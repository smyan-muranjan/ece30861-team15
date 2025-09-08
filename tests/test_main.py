from src import main


def test_main_runs():
    # Since main() is a pass, just check it runs without error
    assert main.main() is None
