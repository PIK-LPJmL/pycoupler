"""Test the LPJmLConfig class."""

from pycoupler.config import read_config


def test_set_spinup_config(test_path):
    """Test the set_config method of the LPJmLCoupler class."""
    # create config for coupled run
    config_spinup = read_config(
        model_path=test_path, file_name="data/lpjml_config.json", spin_up=True
    )

    # set spinup run configuration
    config_spinup.set_spinup(sim_path=test_path)

    # only for global runs = TRUE
    config_spinup.river_routing = False

    # regrid by country - create new (extracted) input files and update config
    config_spinup.regrid(
        sim_path=f"{test_path}/data", country_code="NLD", overwrite_input=False
    )
    assert config_spinup.model_path == test_path
    assert config_spinup.sim_path == test_path
    assert (
        config_spinup.write_restart_filename
        == f"{test_path}/restart/restart_spinup.lpj"
    )
    assert config_spinup.restart_year == 2011
    assert config_spinup.river_routing is False


def test_set_historic_config(test_path):

    # create config for historic run
    config_historic = read_config(
        model_path=test_path, file_name="data/lpjml_config.json"
    )

    # set historic run configuration
    config_historic.set_transient(
        sim_path=test_path,
        sim_name="historic_run",
        dependency="spinup",
        start_year=1901,
        end_year=2000,
    )

    # only for global runs = TRUE
    config_historic.river_routing = False
    config_historic.tillage_type = "read"
    config_historic.residue_treatment = "read_residue_data"
    config_historic.double_harvest = False

    assert config_historic.model_path == test_path
    assert config_historic.sim_path == test_path
    assert config_historic.restart_filename == f"{test_path}/restart/restart_spinup.lpj"
    assert (
        config_historic.write_restart_filename
        == f"{test_path}/restart/restart_historic_run.lpj"
    )
    assert config_historic.restart_year == 2000
    assert config_historic.river_routing is False
    assert config_historic.tillage_type == "read"
    assert config_historic.residue_treatment == "read_residue_data"
    assert config_historic.double_harvest is False


def test_set_coupled_config(test_path):
    """Test the set_config method of the LPJmLCoupler class."""
    # create config for coupled run
    config_coupled = read_config(
        model_path=test_path, file_name="data/lpjml_config.json"
    )

    config_coupled.startgrid = 27410
    config_coupled.endgrid = 27411

    # set coupled run configuration
    config_coupled.set_coupled(
        sim_path=test_path,
        sim_name="coupled_test",
        dependency="historic_run",
        start_year=2001,
        end_year=2050,
        coupled_year=2023,
        coupled_input=["with_tillage"],  # residue_on_field
        coupled_output=[
            "soilc_agr_layer",
            "cftfrac",
            "pft_harvestc",
            "hdate",
            "country",
            "region",
        ],
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
    config_coupled.input.wetdays.name = (
        "CRU_TS4.03/cru_ts4.03.1901.2018.wet.clm"  # noqa
    )
    config_coupled.input.landuse.name = (
        "input_toolbox_30arcmin/cftfrac_1500-2017_64bands_f2o.clm"  # noqa
    )
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
        model_path=test_path, file_name="data/config_coupled_test.json"
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

    # assert that dict config_coupled has the content and structure as
    #   check_config_coupled
    assert config_coupled_dict == check_config_coupled_dict
