"""Test the LPJmLCoupler class."""

import os
import numpy as np
from unittest.mock import patch
from pycoupler.coupler import LPJmLCoupler

from .conftest import get_test_path


@patch.dict(os.environ, {"TEST_PATH": get_test_path(), "TEST_LINE_COUNTER": "0"})
def test_lpjml_coupler(test_path):

    config_coupled_fn = f"{test_path}/data/config_coupled_test.json"
    lpjml_coupler = LPJmLCoupler(config_file=config_coupled_fn)

    """Test the LPJmLCoupler class."""
    inputs = lpjml_coupler.read_input(copy=False)
    outputs = lpjml_coupler.read_historic_output()

    hist_outputs = outputs.copy(deep=True)

    for year in lpjml_coupler.get_sim_years():
        inputs.time.values[0] = np.datetime64(f"{year}-12-31")
        # send input data to lpjml
        lpjml_coupler.send_input(inputs, year)
        # read output data from lpjml

        outputs.time.values[0] = np.datetime64(f"{year}-12-31")
        for name, output in lpjml_coupler.read_output(year).items():
            outputs[name][:] = output[:]

        if year == lpjml_coupler.config.lastyear:
            lpjml_coupler.close()

    # assert that the output is the same as the historic output
    assert np.allclose(outputs["cftfrac"].values, hist_outputs["cftfrac"].values)
    assert not np.allclose(outputs["hdate"].values, hist_outputs["hdate"].values)

    assert not np.allclose(
        outputs["pft_harvestc"].values, hist_outputs["pft_harvestc"].values
    )

    assert not np.allclose(
        outputs["soilc_agr_layer"].values, hist_outputs["soilc_agr_layer"].values
    )
