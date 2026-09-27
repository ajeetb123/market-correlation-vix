"""Tests for the research service and methodology text."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from vixagent.agent import service as service_mod
from vixagent.agent.methodology import TOPICS, methodology_text
from vixagent.agent.service import ResearchService
from vixagent.config import Settings, load_preregistered, load_settings
from vixagent.data.fetch import DataError


@pytest.fixture
def service(panel: pd.DataFrame, small_settings: Settings) -> ResearchService:
    return ResearchService(small_settings, load_preregistered(load_settings()), panel)


def _all_calls(svc: ResearchService) -> list[Any]:
    return [
        lambda: svc.describe_dataset(),
        lambda: svc.correlation_summary("risk", 21, "full"),
        lambda: svc.spike_events("vix", "full"),
        lambda: svc.spike_events("corr", "train", "risk", 21, 2.0),
        lambda: svc.event_study("risk", 21, 2.0, 10, "full", True, "corr_to_vix"),
        lambda: svc.regression("risk", 21, 10, "full", True),
        lambda: svc.oos("risk", 21, 10, 2.0),
        lambda: svc.overfitting_check(),
        lambda: svc.preregistered_spec(),
        lambda: svc.methodology("zscore"),
    ]


def test_every_method_is_strict_json_and_memoized(service: ResearchService) -> None:
    for call in _all_calls(service):
        first = call()
        assert isinstance(first, dict)
        json.dumps(first, allow_nan=False)
        assert call() is first


def test_describe_dataset_fields(service: ResearchService) -> None:
    d = service.describe_dataset()
    assert set(d["tickers"]) == {"risk", "all"}
    assert d["n_trading_days"]["risk"] == 1500
    assert d["rows_dropped_in_alignment"] == {"risk": 0, "all": 0}
    assert "Yahoo Finance" in d["data_source"]


def test_correlation_summary_consistent(service: ResearchService) -> None:
    d = service.correlation_summary("risk", 21, "full")
    assert d["min"] <= d["mean"] <= d["max"]
    assert d["min_date"] <= d["latest_date"]


def test_spike_event_definitions_come_from_settings(
    service: ResearchService, small_settings: Settings
) -> None:
    vix = service.spike_events("vix", "full")
    assert f">= {small_settings.vix_spike.ratio_threshold}" in vix["definition"]
    assert vix["n_events"] >= 3
    corr = service.spike_events("corr", "full", "risk", 21, 2.0)
    assert f"{small_settings.features.z_lookback}-day baseline" in corr["definition"]


def test_dates_are_capped(service: ResearchService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_mod, "DATE_CAP", 3)
    d = service.spike_events("corr", "full", "all", 21, 0.5)
    assert d["n_events"] > 3
    assert len(d["dates"]) == 3
    assert d["truncated"] is True
    es = service.event_study("all", 21, 0.5, 5, "full", False, "corr_to_vix")
    assert es["n_events"] > 3
    assert len(es["event_dates"]) == 3 and len(es["hit_flags"]) == 3
    assert es["truncated"] is True


def test_not_truncated_when_short(service: ResearchService) -> None:
    d = service.spike_events("vix", "full")
    assert d["truncated"] is False
    assert len(d["dates"]) == d["n_events"]


def test_corr_events_require_params(service: ResearchService) -> None:
    with pytest.raises(ValueError):
        service.spike_events("corr", "full")


def test_overfitting_check_shape(service: ResearchService) -> None:
    d = service.overfitting_check()
    assert d["n_combinations"] == 24
    assert d["preregistered"]["window"] == 21
    assert set(d["preregistered"]) == {
        "group",
        "window",
        "z_threshold",
        "horizon",
        "train_lift",
        "train_n_events",
        "test_lift",
        "test_n_events",
    }


def test_preregistered_spec(service: ResearchService) -> None:
    d = service.preregistered_spec()
    assert d["primary"]["horizon"] == 10
    assert "prereg-freeze" in d["frozen"]


def test_from_cache_without_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"Run `vixagent pull` first"):
        ResearchService.from_cache(root=tmp_path)


@pytest.mark.parametrize("topic", TOPICS)
def test_methodology_texts(topic: str) -> None:
    s = load_settings()
    text = methodology_text(topic, s, load_preregistered(s))
    assert text.strip()
    assert len(text.split()) <= 200
    assert "{" not in text.replace("{t-1}", "").replace("{t+h}", "").replace("{t-5}", "")


def test_vix_spike_text_uses_threshold() -> None:
    s = load_settings()
    text = methodology_text("vix_spike", s, load_preregistered(s))
    assert str(s.vix_spike.ratio_threshold) in text


def test_unknown_topic_raises() -> None:
    s = load_settings()
    with pytest.raises(KeyError):
        methodology_text("astrology", s, load_preregistered(s))
