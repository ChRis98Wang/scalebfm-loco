"""Real-Kit smoke for cleanup after OnlineTargetPanel setup fails.

Run with the normal ``play.py`` online arguments and ``--viz kit``.  The
parent process owns the 180 second process-group deadline.  This driver exits
nonzero by design after proving that the injected setup exception propagated
through both panel and environment cleanup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


class InjectedSetupFailure(RuntimeError):
    """Expected failure after the native online window has been built."""


REPORT_VALUE = os.environ.get("BFM_ONLINE_SMOKE_REPORT", "")
if not REPORT_VALUE:
    raise ValueError("BFM_ONLINE_SMOKE_REPORT must name a JSON output file")

REPORT_PATH = Path(REPORT_VALUE).expanduser().resolve()
report = {
    "result": "RUNNING",
    "started_monotonic_s": time.monotonic(),
    "argv": list(sys.argv),
    "environment_created": False,
    "injection_installed_after_environment": False,
    "injection_reached": False,
    "panel_window_created_before_failure": False,
    "panel_close_attempts": 0,
    "panel_close_successes": 0,
    "environment_close_attempts": 0,
    "environment_close_successes": 0,
    "app_close_attempted": False,
    "app_close_success": False,
}


def write_report() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


scripts = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"
sys.path.insert(0, str(scripts))

player = None
app = None
gym_module = None
panel_class = None
original_gym_make = None
original_panel_close = None
original_build_markers = None

try:
    # Default run_name loads play.py and launches its AppLauncher, but does not
    # enter play.py's __main__ block or close the app behind this driver.
    player = runpy.run_path(str(scripts / "play.py"))
    app = player["simulation_app"]

    import gymnasium as gym_module
    from online_ui import OnlineTargetPanel as panel_class

    original_gym_make = gym_module.make
    original_panel_close = panel_class.__dict__["close"]
    original_build_markers = panel_class.__dict__["_build_markers"]

    def observed_panel_close(panel) -> None:
        report["panel_close_attempts"] += 1
        try:
            original_panel_close(panel)
        except BaseException as error:
            report["panel_close_error"] = f"{type(error).__name__}: {error}"
            raise
        report["panel_close_successes"] += 1
        report["panel_callback_detached"] = True  # close returned past its unconditional detach call
        report["panel_window_close_completed"] = True
        report["panel_subscription_released"] = panel._subscription is None
        report["panel_marker_released"] = panel._marker is None
        try:
            report["panel_window_hidden"] = panel.window is not None and not panel.window.visible
        except BaseException as error:
            # Some native Window bindings reject reads after destroy; successful
            # close still proves detach/hide/destroy completed without raising.
            report["panel_window_post_destroy"] = f"unreadable: {type(error).__name__}: {error}"

    panel_class.close = observed_panel_close

    def fail_build_markers(panel):
        report["injection_reached"] = True
        report["panel_window_created_before_failure"] = panel.window is not None
        error = InjectedSetupFailure("injected online marker setup failure")
        report["injected_exception_type"] = type(error).__name__
        report["injected_exception"] = str(error)
        raise error

    def observed_gym_make(*args, **kwargs):
        env = original_gym_make(*args, **kwargs)
        report["environment_created"] = True
        report["environment_created_monotonic_s"] = time.monotonic()
        original_env_close = env.close

        def observed_env_close(*close_args, **close_kwargs):
            report["environment_close_attempts"] += 1
            try:
                result = original_env_close(*close_args, **close_kwargs)
            except BaseException as error:
                report["environment_close_error"] = f"{type(error).__name__}: {error}"
                raise
            report["environment_close_successes"] += 1
            return result

        env.close = observed_env_close
        # Install the sole failure injection only after the real environment
        # exists; play.py registers its environment callback after make returns.
        panel_class._build_markers = fail_build_markers
        report["injection_installed_after_environment"] = True
        report["injection_installed_monotonic_s"] = time.monotonic()
        return env

    gym_module.make = observed_gym_make
    player["main"]()
    raise AssertionError("play.py returned without the injected setup failure")
except BaseException as error:
    report["observed_exception_type"] = type(error).__name__
    report["observed_exception"] = str(error)
    report["traceback"] = traceback.format_exc()
    expected_exception = isinstance(error, InjectedSetupFailure) or (
        isinstance(error, SystemExit) and error.code not in (None, 0)
    )
    cleanup_ok = (
        report["environment_created"]
        and report["injection_installed_after_environment"]
        and report["injection_reached"]
        and report["panel_window_created_before_failure"]
        and report["panel_close_attempts"] >= 1
        and report["panel_close_successes"] >= 1
        and report.get("panel_callback_detached") is True
        and report.get("panel_window_close_completed") is True
        and report["environment_close_attempts"] >= 1
        and report["environment_close_successes"] >= 1
    )
    report["result"] = "EXPECTED_FAILURE" if expected_exception and cleanup_ok else "FAIL"
    report["request_exit_monotonic_s"] = time.monotonic()
    write_report()
    traceback.print_exc()
    print(
        f"[BFM ONLINE SETUP FAILURE] {report['result']} report={REPORT_PATH}",
        flush=True,
    )
    raise
finally:
    active_exception = sys.exc_info()[0] is not None
    if gym_module is not None and original_gym_make is not None:
        gym_module.make = original_gym_make
    if panel_class is not None:
        if original_build_markers is not None:
            panel_class._build_markers = original_build_markers
        if original_panel_close is not None:
            panel_class.close = original_panel_close
    if app is not None:
        report["app_close_attempted"] = True
        report.setdefault("request_exit_monotonic_s", time.monotonic())
        write_report()
        try:
            app.close(exit_code=1)
        except BaseException as error:
            report["result"] = "FAIL"
            report["app_close_error"] = f"{type(error).__name__}: {error}"
            try:
                write_report()
            except BaseException:
                traceback.print_exc()
            if not active_exception:
                raise
        else:
            report["app_close_success"] = True
            write_report()
