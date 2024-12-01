"""Test the LPJmLConfig class."""

from pycoupler.data import read_data


def test_read_data(test_path):
    """Test the set_config method of the LPJmLCoupler class."""
    # create config for coupled run
    tillage_data = read_data(
        file_name=f"{test_path}/data/input/with_tillage.nc", var_name="with_tillage"
    )
    assert tillage_data.__class__.__name__ == "LPJmLData"

    tillage_data = read_data(file_name=f"{test_path}/data/input/with_tillage.nc")
    assert tillage_data.__class__.__name__ == "LPJmLDataSet"
