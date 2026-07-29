import json
import logging

from cognition.utils.logging import configure_file_logging, get_logger, log_with_fields


def test_module_logger_writes_to_file_only_after_configure(tmp_path):
    """A logger created at import time (like downloader's or
    paper_trader's) must start writing to a file the moment ANY CLI
    configures file logging in this process — even though the logger
    itself was created long before the log directory was known.
    """
    logger = get_logger("some_module_created_early")
    log_with_fields(logger, logging.INFO, "before configure", n=1)

    path = configure_file_logging(tmp_path, "paper_trader")
    log_with_fields(logger, logging.INFO, "after configure", n=2)

    assert path == tmp_path / "paper_trader.log"
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    messages = [l["message"] for l in lines]
    assert "after configure" in messages
    assert "before configure" not in messages  # no handler existed yet


def test_paper_trader_module_logger_reaches_the_healthcheck_file(tmp_path):
    """Docker's healthcheck (docker-compose.yml) watches logs/paper_trader.log
    for a recent mtime. cognition.paper.trader creates its logger at
    import time via `logger = get_logger("paper_trader")`, so this proves
    that exact logger reaches that exact file once paper_trade.py's CLI
    calls configure_file_logging(log_dir, "paper_trader") at startup.
    """
    from cognition.paper.trader import logger as paper_trader_logger

    path = configure_file_logging(tmp_path, "paper_trader")
    log_with_fields(paper_trader_logger, logging.INFO, "heartbeat")

    assert path.exists()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(l["message"] == "heartbeat" and l["logger"] == "paper_trader" for l in lines)


def test_other_module_loggers_also_land_in_the_same_run_file(tmp_path):
    """Every cognition.* module active during a run should land in that
    run's file, not just the CLI's own top-level logger — otherwise
    download_data.log silently omits the downloader's own progress lines.
    """
    from cognition.data.bybit_client import logger as bybit_logger
    from cognition.data.downloader import logger as downloader_logger

    configure_file_logging(tmp_path, "download_data")
    log_with_fields(downloader_logger, logging.INFO, "Saved candles", new_candles=5)
    log_with_fields(bybit_logger, logging.INFO, "Bybit client initialized")

    path = tmp_path / "download_data.log"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    loggers_seen = {l["logger"] for l in lines}
    assert {"downloader", "bybit_client"} <= loggers_seen


def test_configure_file_logging_does_not_accumulate_handlers(tmp_path):
    """Calling configure_file_logging repeatedly (e.g. once per test in
    this same process) must replace, not stack, the file handler —
    otherwise later runs would keep writing into earlier runs' files too.
    """
    configure_file_logging(tmp_path, "first_run")
    configure_file_logging(tmp_path, "second_run")

    logger = get_logger("some_other_module")
    log_with_fields(logger, logging.INFO, "only second run should see this")

    first_log = tmp_path / "first_run.log"
    second_log = tmp_path / "second_run.log"
    assert not first_log.exists() or "only second run" not in first_log.read_text()
    assert "only second run" in second_log.read_text()


def test_console_output_is_unaffected_by_file_configuration(tmp_path, capsys):
    logger = get_logger("console_check")
    configure_file_logging(tmp_path, "console_check_run")
    log_with_fields(logger, logging.INFO, "printed once")

    captured = capsys.readouterr()
    assert captured.out.count("printed once") == 1
    assert captured.err.count("printed once") == 0
