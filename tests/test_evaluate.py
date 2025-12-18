"""
This is a test module for our evaluation code.
"""

import pytest

from automaldesc import evaluate


def test_metric() -> None:
    predictions = [
        '{"language": "python",     "label": 0, "summary": "Idea 1"}',
        "malformed",
        '{"language": "python",     "label": 1, "summary": "Idea 2"}',
    ]
    references = [
        '{"language": "python",     "label": 0, "summary": "Idea 1"}',
        '{"language": "javascript", "label": 0, "summary": "Idea 7"}',
        '{"language": "python",     "label": 0, "summary": "No idea"}',
    ]
    metric = evaluate.AutoMalDescScore()
    metric.add_batch(predictions=predictions, references=references)
    res = metric.compute()
    assert res["json-err-fraction"] == pytest.approx(1 / 3)
    assert res["acc-language"] == pytest.approx(2 / 3)
    assert res["acc-label"] == pytest.approx(1 / 3)
    assert res["acc-summary"] == pytest.approx(1 / 3)
    assert res["acc-python-language"] == pytest.approx(2 / 2)

    # We should warn about malformed references.
    with pytest.raises(ValueError):
        metric = evaluate.AutoMalDescScore()
        metric.add_batch(
            predictions=['{"language": "python",     "label": 0, "summary": "Idea 1"}'],
            references=["malformed"],
        )
