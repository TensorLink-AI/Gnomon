def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens", action="store_true", default=False,
        help="Rewrite golden artifact files from the current runtime output",
    )
