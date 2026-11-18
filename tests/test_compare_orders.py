import math

from quic_wc.pilot.compare_orders import bytes_for_error, family_of, pareto, saving_text


def test_pareto_and_interpolation():
    front = pareto([(0.02, 0.4), (0.01, 0.5), (0.04, 0.3), (0.03, 0.45)])
    assert front == [(0.01, 0.5), (0.02, 0.4), (0.04, 0.3)]
    value, rel = bytes_for_error(front, 0.4)
    assert rel == "=" and math.isclose(value, 0.02)
    value, rel = bytes_for_error(front, math.sqrt(0.5 * 0.4))  # log-midpoint of the first segment
    assert rel == "=" and math.isclose(value, math.sqrt(0.01 * 0.02))


def test_bounds_outside_the_reference_curve():
    front = [(0.01, 0.5), (0.04, 0.3)]
    assert bytes_for_error(front, 0.6) == (0.01, "<")  # variant worse than the cheapest reference point
    assert bytes_for_error(front, 0.2) == (0.04, ">")  # reference never reaches the variant's error
    _, text, saving, rel = saving_text((0.01, "<"), 0.02)
    assert rel == ">" and math.isclose(saving, 0.5) and text == ">50.0%"
    assert saving_text(None, 0.02)[2] is None


def test_family_of_strips_single_spec_suffix():
    assert family_of({"family": "k1eq_rand1_L14"}) == "k1eq_rand1"
    assert family_of({"family": "k12_stride"}) == "k12_stride"
