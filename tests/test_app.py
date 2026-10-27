from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "dashboard.py")


@pytest.mark.parametrize("page", ["Portfolio", "Loan Application", "Model & Fairness"])
def test_pages_render(page):
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
