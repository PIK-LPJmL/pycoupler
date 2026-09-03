"""Test the LPJmLCoupler class."""

import pytest


def test_lpjml_coupler(model_path, sim_path, lpjml_coupler):
    inputs = lpjml_coupler.read_input(copy=False)
    hist_outputs = lpjml_coupler.read_historic_output()

    for year in lpjml_coupler.get_sim_years():
        # send input data to lpjml
        lpjml_coupler.send_input(inputs, year)
        # read output data from lpjml
        output = lpjml_coupler.read_output(year)

        # TODO: These assertions are wrong and need to be checked against the mocked socket values in a later version
        # assert that the output is the same as the historic output
        # assert np.allclose(a=output["cftfrac"].values, hist_outputs["cftfrac"].values)
        # assert not np.allclose(output["hdate"].values, hist_outputs["hdate"].values)

        # assert not np.allclose(
        #     output["pft_harvestc"].values, hist_outputs["pft_harvestc"].values
        # )

        # assert not np.allclose(
        #     output["soilc_agr_layer"].values, hist_outputs["soilc_agr_layer"].values
        # )

        if year == lpjml_coupler.config.lastyear:
            lpjml_coupler.close()

    assert "_channel" not in lpjml_coupler.__getstate__()
    assert lpjml_coupler.ncell == 2
    assert [year for year in lpjml_coupler.get_cells()] == [27410, 27411]
    assert lpjml_coupler.historic_years == []
    assert lpjml_coupler.sim_years == []
    assert lpjml_coupler.coupled_years == []
    assert [year for year in lpjml_coupler.get_coupled_years()] == []


def test_lpjml_coupler_repr(model_path, sim_path, lpjml_coupler):
    assert (
        repr(lpjml_coupler)
        == f"""<pycoupler.LPJmLCoupler>
Simulation:  (version: 3, localhost:<none>)
  * sim_year   2022
  * ncell      2
  * ninput     1
Configuration:
  Settings:      lpjml v5.8
    (general)
    * sim_name   coupled_test
    * firstyear  2001
    * lastyear   2050
    * startgrid  27410
    * endgrid    27411
    * landuse    yes
    (changed)
    * model_path           {model_path}
    * sim_path             {sim_path}
    * outputyear           2022
    * output_metafile      True
    * write_restart        False
    * nspinup              0
    * float_grid           True
    * restart_filename     restart/restart_historic_run.lpj
    * outputyear           2022
    * radiation            cloudiness
    * fix_co2              True
    * fix_co2_year         2018
    * fix_climate          True
    * fix_climate_cycle    11
    * fix_climate_year     2013
    * river_routing        False
    * tillage_type         read
    * residue_treatment    fixed_residue_remove
    * double_harvest       False
    * intercrop            True
  Coupled model:        copan:CORE
    * start_coupling    2023
    * input (coupled)   ['with_tillage']
    * output (coupled)  ['grid', 'pft_harvestc', 'cftfrac', 'soilc_agr_layer', 'hdate', 'country', 'region']
  """  # noqa
    )


def test_lpjml_coupler_codes_name(lpjml_coupler):
    lpjml_coupler.code_to_name(to_iso_alpha_3=False)
    assert lpjml_coupler.country[0].item() == "Germany"


def test_lpjml_coupler_codes_iso(lpjml_coupler):
    lpjml_coupler.code_to_name(to_iso_alpha_3=True)
    assert lpjml_coupler.country[0].item() == "DEU"


# Test all period combination cases (data period is 2000 to 2022)
@pytest.mark.parametrize(
    "start_year,end_year",
    [
        (2005, 2015),
        (1980, 1998),
        (2024, 2025),
        (1998, 2025),
        (1998, 2020),
        (2020, 2025),
        (None, 2024),
        (1998, None),
        (None, None),
        pytest.param(
            2025,
            1998,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_lpjml_coupler_copy_input_(test_path, lpjml_coupler, start_year, end_year):
    assert lpjml_coupler._copy_input(start_year, end_year) == "tested"
