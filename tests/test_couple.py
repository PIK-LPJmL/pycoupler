"""Test the LPJmLCoupler class."""

import os
import numpy as np
from unittest.mock import patch
from copy import deepcopy
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

    assert "_channel" not in lpjml_coupler.__getstate__()

    assert lpjml_coupler.ncell == 2
    assert [year for year in lpjml_coupler.get_cells()] == [27410, 27411]
    assert lpjml_coupler.historic_years == []
    assert lpjml_coupler.sim_years == []
    assert lpjml_coupler.coupled_years == []
    assert [year for year in lpjml_coupler.get_coupled_years()] == []

    second_coupler = deepcopy(lpjml_coupler)
    lpjml_coupler.code_to_name(to_iso_alpha_3=False)
    assert lpjml_coupler.country[0].item() == "Germany"
    second_coupler.code_to_name(to_iso_alpha_3=True)
    assert second_coupler.country[0].item() == "DEU"

    assert (
        repr(lpjml_coupler)
        == f"<pycoupler.LPJmLCoupler>\nSimulation:  (version: 3, localhost:<none>)\n  * sim_year   2050\n  * ncell      2\n  * ninput     1\nConfiguration:\n  Settings:      lpjml v5.8\n    (general)\n    * sim_name   coupled_test\n    * firstyear  2001\n    * lastyear   2050\n    * startgrid  27410\n    * endgrid    27411\n    * landuse    yes\n    (changed)\n    * model_path           LPJmL_internal\n    * sim_path             {test_path}/data/\n    * outputyear           2022\n    * output_metafile      True\n    * write_restart        False\n    * nspinup              0\n    * float_grid           True\n    * restart_filename     restart/restart_historic_run.lpj\n    * outputyear           2022\n    * radiation            cloudiness\n    * fix_co2              True\n    * fix_co2_year         2018\n    * fix_climate          True\n    * fix_climate_cycle    11\n    * fix_climate_year     2013\n    * river_routing        False\n    * tillage_type         read\n    * residue_treatment    fixed_residue_remove\n    * double_harvest       False\n    * intercrop            True\n    * sim_path             {test_path}/data/\n  Coupled model:        copan:CORE\n    * start_coupling    2023\n    * input (coupled)   ['with_tillage']\n    * output (coupled)  ['grid', 'pft_harvestc', 'cftfrac', 'soilc_agr_layer', 'hdate', 'country', 'region']\n  "  # noqa
    )


@patch.dict(os.environ, {"TEST_PATH": get_test_path(), "TEST_LINE_COUNTER": "0"})
def test_copy_input(test_path):

    config_coupled_fn = f"{test_path}/data/config_coupled_test.json"
    lpjml_coupler = LPJmLCoupler(config_file=config_coupled_fn)

    inputs = lpjml_coupler.read_input(copy=False)

    assert lpjml_coupler._copy_input(start_year=2022, end_year=2022) == "tested"
