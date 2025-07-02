import os
import pytest
import json


def get_test_path():
    """Fixture for the test path."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def test_path():
    """Fixture for the test path."""
    return get_test_path()


@pytest.fixture()
def mock_venv(tmp_path_factory):
    venv = tmp_path_factory.mktemp("venv")
    (venv / "bin").mkdir()
    (venv / "bin" / "python").touch()
    return venv


@pytest.fixture()
def sim_path(tmp_path_factory):
    sim_fn = tmp_path_factory.mktemp("sim")
    return sim_fn


@pytest.fixture()
def model_path(tmp_path_factory):
    model_fn = tmp_path_factory.mktemp("model")
    return model_fn


@pytest.fixture()
def config_coupled(sim_path, model_path, test_path):
    new_config = sim_path / "config_coupled.json"
    with open(f"{test_path}/data/config_coupled_test.json") as conf:
        conf_d = json.load(conf)
        conf_d["model_path"] = str(model_path)
        conf_d["sim_path"] = str(sim_path)
        with new_config.open("w") as f:
            json.dump(conf_d, f)
            return str(new_config)


def pytest_configure(config):
    import sys

    sys._called_from_test = True


def pytest_unconfigure(config):
    import sys  # This was missing from the manual

    del sys._called_from_test
