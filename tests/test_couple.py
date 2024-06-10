"""Test the LPJmLCoupler class."""
import os
import numpy as np
import pytest
from pycoupler.config import read_config
from pycoupler.coupler import LPJmLCoupler



@pytest.fixture
def test_path():
    """Fixture for the test path."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def lpjml_coupler(test_path):
    """Fixture for the LPJmLCoupler class."""
    config_coupled_fn = f"{test_path}/data/config_coupled_test.json"
    return LPJmLCoupler(config_file=config_coupled_fn)


def test_lpjml_coupler(lpjml_coupler):
    """Test the LPJmLCoupler class."""
    inputs = lpjml_coupler.read_input(copy=False)
    outputs = lpjml_coupler.read_historic_output()

    hist_outputs = outputs.copy(deep = True)

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
    assert np.allclose(
        outputs["cftfrac"].values, hist_outputs["cftfrac"].values
    )
    assert not np.allclose(
        outputs["hdate"].values, hist_outputs["hdate"].values
    )

    assert not np.allclose(
        outputs["pft_harvestc"].values, hist_outputs["pft_harvestc"].values
    )

    assert not np.allclose(
        outputs["soilc_agr_layer"].values, hist_outputs["soilc_agr_layer"].values
    )


def test_set_config(test_path):
    """Test the set_config method of the LPJmLCoupler class."""
    # create config for coupled run
    config_coupled = read_config(
        model_path=test_path, file_name="data/lpjml_config.json"
    )

    config_coupled.startgrid = 27410
    config_coupled.endgrid = 27411

    # set coupled run configuration
    config_coupled.set_coupled(
        test_path,
        sim_name="coupled_test",
        dependency="historic_run",
        start_year=2001, end_year=2050,
        coupled_year=2023,
        coupled_input=["with_tillage"],  # residue_on_field
        coupled_output=[
            "soilc_agr_layer",
            "cftfrac",
            "pft_harvestc",
            "hdate",
            "country",
            "region"
        ]
    )

    # only for single cells runs
    config_coupled.outputyear = 2022

    # set more recent input files
    config_coupled.radiation = "cloudiness"
    config_coupled.input.temp.name = "CRU_TS4.03/cru_ts4.03.1901.2018.tmp.clm"
    config_coupled.input.prec.name = "CRU_TS4.03/cru_ts4.03.1901.2018.pre.clm"
    config_coupled.input.cloud.name = "CRU_TS4.03/cru_ts4.03.1901.2018.cld.clm"
    config_coupled.fix_co2 = True
    config_coupled.fix_co2_year = 2018
    config_coupled.input.co2.name = "input_VERSION2/co2_1841-2018.dat"
    config_coupled.input.wetdays.name = "CRU_TS4.03/cru_ts4.03.1901.2018.wet.clm" # noqa
    config_coupled.input.landuse.name = "input_toolbox_30arcmin/cftfrac_1500-2017_64bands_f2o.clm" # noqa
    config_coupled.fix_climate = True
    config_coupled.fix_climate_cycle = 11
    config_coupled.fix_climate_year = 2013

    # only for global runs = TRUE
    config_coupled.river_routing = False
    config_coupled.tillage_type = "read"
    config_coupled.residue_treatment = "fixed_residue_remove"
    config_coupled.double_harvest = False
    config_coupled.intercrop = True

    # create config for coupled run
    check_config_coupled = read_config(
        model_path = test_path, file_name="data/config_coupled_test.json"
    )
    # update with actual output path (test directory)
    check_config_coupled.set_outputpath(f"{test_path}/output/coupled_test")

    # align both config objects
    check_config_coupled.restart_filename = config_coupled.restart_filename
    check_config_coupled.sim_path = config_coupled.sim_path

    # delete tracking enty changed from dict for comparison
    config_coupled_dict = config_coupled.to_dict()
    check_config_coupled_dict = check_config_coupled.to_dict()
    del config_coupled_dict["changed"]
    del check_config_coupled_dict["changed"]

    # assert that dict config_coupled has the content and structure as check_config_coupled
    assert config_coupled_dict == check_config_coupled_dict
