import pytest

from tennis_analyze.court_model import court_lines


def test_court_has_seven_lines_with_spec_dimensions():
    lines = court_lines()
    assert len(lines) == 7
    xs = {p[0] for line in lines for p in line}
    ys = {p[1] for line in lines for p in line}
    assert min(xs) == 0 and max(xs) == 8.23
    assert min(ys) == 0 and max(ys) == 23.77


def test_service_lines_are_6_40m_from_net():
    ys = {p[1] for line in court_lines() for p in line}
    assert any(y == pytest.approx(23.77 / 2 - 6.40) for y in ys)  # 5.485
    assert any(y == pytest.approx(23.77 / 2 + 6.40) for y in ys)  # 18.285
