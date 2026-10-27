"""Configuration validation without ROS dependencies."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ros2_sid.flight_config import (
    NULLABLE_CHECKS, load_straight_flight_config, parse_rc_timeout, resolve_rc_timeout,
)

CONFIG = Path(__file__).parents[1] / "config/surface_excitation/straight_flight_sitl.json"


@pytest.mark.parametrize("config,override,expected", [
    (None, {}, 0.5), ({}, {}, 0.5), ({"rc_timeout_s": None}, {}, None),
    ({"rc_timeout_s": 1}, {}, 1),
    ({"rc_timeout_s": None}, {"rc_timeout": 0.2}, 0.2),
    ({"rc_timeout_s": 1}, {"rc_timeout": None}, None),
])
def test_rc_precedence(config, override, expected):
    args = SimpleNamespace(**override)
    assert resolve_rc_timeout(args, config) == expected
    assert args.rc_timeout == expected
    if config is not None:
        assert config["rc_timeout_s"] == expected


def test_rc_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc-timeout", type=parse_rc_timeout, default=argparse.SUPPRESS)
    assert not hasattr(parser.parse_args([]), "rc_timeout")
    assert parser.parse_args(["--rc-timeout", "null"]).rc_timeout is None
    assert parser.parse_args(["--rc-timeout", "0.5"]).rc_timeout == 0.5
    for value in ("0", "-1", "2.1", "nan", "inf", "invalid"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--rc-timeout", value])


def test_packaged_rc_disabled():
    assert load_straight_flight_config(CONFIG)["rc_timeout_s"] is None


def load_modified(tmp_path, **updates):
    config = json.loads(CONFIG.read_text())
    config.update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return load_straight_flight_config(path)


@pytest.mark.parametrize("key", NULLABLE_CHECKS)
def test_nullable_checks(tmp_path, key):
    assert load_modified(tmp_path, **{key: None})[key] is None
    for value in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match=key):
            load_modified(tmp_path, **{key: value})


@pytest.mark.parametrize("key", [
    "altitude_relative_m", "airspeed_m_s", "heading_accel_m_s2", "dwell_s",
    "setup_timeout_s", "recovery_timeout_s", "telemetry_timeout_s", "state_timeout_s",
    "climb_rate_m_s",
])
def test_required_numbers_reject_null(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        load_modified(tmp_path, **{key: None})


def test_independent_airspeed_bounds(tmp_path):
    load_modified(tmp_path, min_airspeed_m_s=None, airspeed_m_s=10)
    load_modified(tmp_path, max_airspeed_m_s=None, airspeed_m_s=40)
    load_modified(tmp_path, min_airspeed_m_s=None, max_airspeed_m_s=None)
    with pytest.raises(ValueError, match="envelope"):
        load_modified(tmp_path, min_airspeed_m_s=None, airspeed_m_s=40)
    with pytest.raises(ValueError, match="envelope"):
        load_modified(tmp_path, max_airspeed_m_s=None, airspeed_m_s=10)


def test_missing_rc_default_and_required_keys(tmp_path):
    config = json.loads(CONFIG.read_text())
    del config["rc_timeout_s"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_straight_flight_config(path)["rc_timeout_s"] == 0.5
    del config["max_bank_deg"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="max_bank_deg"):
        load_straight_flight_config(path)
