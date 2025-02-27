from dashboard import get_prediction


def test_get_prediction_accord():
    assert get_prediction(100038) == (0.379, 'Approved')


def test_get_prediction_refus():
    assert get_prediction(436394) == (0.671, 'Rejected')