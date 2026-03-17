import pytest


@pytest.fixture
def client():
    pytest.importorskip("shapely")
    from src.terra_ugla.app import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
