"""真实延迟基准的统计口径。"""

from scripts.bench_real_e2e import _percentiles


def test_p95_interpolates_tail_instead_of_ignoring_slowest_sample() -> None:
    values = [100.0, 200.0, 300.0, 400.0]
    result = _percentiles(values)
    assert result["p50"] == 250.0
    assert result["p95"] == 385.0
