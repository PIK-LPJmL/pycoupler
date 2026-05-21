import os
from pathlib import Path
import pytest
from pycoupler.coupler import LPJmLCoupler
import json
import shutil


@pytest.fixture
def test_path():
    """Fixture for the test path."""
    return Path(__file__).absolute().parent


@pytest.fixture
def lpjml_coupler(config_coupled_file: Path):
    os.environ["TEST_LINE_COUNTER"] = "0"
    # Using yield enables safe teardown of the fixture
    # (see https://docs.pytest.org/en/stable/how-to/fixtures.html#safe-teardowns)
    yield LPJmLCoupler(config_file=config_coupled_file)
    # Reset test line env variable
    del os.environ["TEST_LINE_COUNTER"]


@pytest.fixture()
def sim_path(tmp_path_factory):
    sim_fn = tmp_path_factory.mktemp("sim")
    return sim_fn


@pytest.fixture()
def output_path(sim_path: Path, test_path: Path):
    top_fn = sim_path / "output"
    top_fn.mkdir()
    return Path(
        shutil.copytree(test_path / "data/output/coupled_test", top_fn / "coupled_test")
    )


@pytest.fixture()
def sim_inputs(sim_path: Path, test_path: Path):
    return Path(shutil.copytree(test_path / "data/input", sim_path / "input"))


@pytest.fixture()
def model_path(tmp_path_factory):
    model_fn = tmp_path_factory.mktemp("model")
    return model_fn


def outputpath_helper(output_dict, path):
    output_dict["file"]["name"] = output_dict["file"]["name"].replace(
        "output/", path + "/"
    )
    return output_dict


@pytest.fixture()
def lpjml_config(
    sim_path: Path,
    model_path: Path,
    test_path: Path,
    sim_inputs: Path,
    output_path: Path,
):
    new_config = model_path / "lpjml_config.json"
    with (test_path / "data/lpjml_config.json").open("r") as conf:
        conf_d = json.load(conf)
        conf_d["model_path"] = str(model_path)
        conf_d["sim_path"] = str(sim_path)
        conf_d["inpath"] = str(sim_inputs)
        conf_d["output"] = [
            outputpath_helper(out, str(output_path)) for out in conf_d["output"]
        ]
        with new_config.open("w") as f:
            json.dump(conf_d, f)
            return str(new_config)


@pytest.fixture()
def config_coupled_file(
    sim_path: Path,
    model_path: Path,
    test_path: Path,
    sim_inputs: Path,
    output_path: Path,
):
    new_config = sim_path / "config_coupled.json"
    with (test_path / "data/config_coupled_test.json").open("r") as conf:
        conf_d = json.load(conf)
        conf_d["model_path"] = str(model_path)
        conf_d["sim_path"] = str(sim_path)
        conf_d["inpath"] = str(sim_inputs)
        conf_d["output"] = [
            outputpath_helper(out, str(output_path)) for out in conf_d["output"]
        ]
        with new_config.open("w") as f:
            json.dump(conf_d, f)
            return str(new_config)


def pytest_configure(config):
    import sys

    sys._called_from_test = True
    os.environ["TEST_PATH"] = os.path.dirname(os.path.abspath(__file__))
    os.environ["TEST_LINE_COUNTER"] = "0"


def pytest_unconfigure(config):
    import sys  # This was missing from the manual

    del sys._called_from_test
