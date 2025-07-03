"""Test the LPJmLData class."""

import numpy as np

from pycoupler.data import (
    read_data,
    read_meta,
    read_header,
    get_headersize,
    LPJmLInputType,
    append_to_dict,
)


def test_read_data(test_path):
    """Test the set_config method of the LPJmLCoupler class."""
    # create config for coupled run
    tillage_data = read_data(
        file_name=f"{test_path}/data/input/with_tillage.nc", var_name="with_tillage"
    )
    assert tillage_data.__class__.__name__ == "LPJmLData"

    tillage_data.add_meta(
        read_meta(f"{test_path}/data/input/with_tillage.nc.json")
    )  # noqa
    assert tillage_data.attrs["comment"] == "check"

    tillage_data = read_data(
        file_name=f"{test_path}/data/input/with_tillage.nc"
    )  # noqa
    assert tillage_data.__class__.__name__ == "LPJmLDataSet"


def test_dataset(test_path):
    """Test the set_config method of the LPJmLCoupler class."""
    tillage_data = read_data(
        file_name=f"{test_path}/data/input/with_tillage.nc"
    )  # noqa
    data_dict = tillage_data.to_dict("lpjmldata")
    assert list(data_dict.keys()) == ["with_tillage"]


def test_get_neighbourhood(lpjml_coupler):
    neighbourhood = lpjml_coupler.grid.get_neighbourhood().values

    test_neighbours = np.array(
        [
            [27411, -9999, -9999, -9999, -9999, -9999, -9999, -9999],
            [27410, -9999, -9999, -9999, -9999, -9999, -9999, -9999],
        ]
    )

    assert np.array_equal(neighbourhood, test_neighbours)


def test_metadata(test_path):

    meta_soil = read_meta(
        f"{test_path}/data/output/coupled_test/soilc_agr_layer.nc4.json"
    )

    assert (
        repr(meta_soil)
        == "<pycoupler.LPJmLMetaData>\n  * sim_name      coupled_test\n  * source        LPJmL C Version 5.8.1\n  * history       /p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json\n  * variable      soilc_agr_layer\n  * long_name     total soil carbon density agricultural stands in layer\n  * unit          gC/m2\n  * nbands        5\n  * band_names    [200.0, 500.0, 1000.0, 2000.0, 3000.0]\n  * nyear         29\n  * firstyear     2022\n  * lastyear      2050\n  * cellsize_lon  0.5\n  * cellsize_lat  0.5\n  * ncell         2\n  * firstcell     27410\n  * datatype      float\n  * scalar        1.0\n  * nstep         1\n  * timestep      1\n  * order         cellseq\n  * bigendian     False\n  * format        cdf\n  * grid          {'filename': 'grid.nc4.json', 'format': 'meta'}\n  * ref_area      {'filename': 'terr_area.nc4.json', 'format': 'meta'}\n  * filename      soilc_agr_layer.nc4"  # noqa
    )

    meta_soil_dict = meta_soil.to_dict()

    check_meta_soil_dict = {
        "sim_name": "coupled_test",
        "source": "LPJmL C Version 5.8.1",
        "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
        "variable": "soilc_agr_layer",
        "firstcell": 27410,
        "ncell": 2,
        "cellsize_lon": 0.5,
        "cellsize_lat": 0.5,
        "nstep": 1,
        "timestep": 1,
        "nbands": 5,
        "band_names": [200.0, 500.0, 1000.0, 2000.0, 3000.0],
        "long_name": "total soil carbon density agricultural stands in layer",
        "unit": "gC/m2",
        "firstyear": 2022,
        "lastyear": 2050,
        "nyear": 29,
        "datatype": "float",
        "scalar": 1.0,
        "order": "cellseq",
        "bigendian": False,
        "format": "cdf",
        "grid": {"filename": "grid.nc4.json", "format": "meta"},
        "ref_area": {"filename": "terr_area.nc4.json", "format": "meta"},
        "filename": "soilc_agr_layer.nc4",
    }

    assert meta_soil_dict == check_meta_soil_dict


def test_read_header(test_path):

    soil_header = read_header(
        f"{test_path}/data/input/soil_netherlands.clm", to_dict=True
    )
    check_soil_header = {
        "name": "LPJSOIL",
        "header": {
            "version": 3,
            "order": 1,
            "firstyear": 1901,
            "nyear": 1,
            "firstcell": 0,
            "ncell": 21,
            "nbands": 1,
            "cellsize_lon": 0.5,
            "scalar": 1.0,
            "cellsize_lat": 0.5,
            "datatype": 0,
            "nstep": 1,
            "timestep": 1,
        },
        "endian": "little",
    }
    assert soil_header == check_soil_header

    append_to_dict(soil_header, {"test": "check"})
    assert soil_header["test"] == "check"

    grid_header = read_header(f"{test_path}/data/input/coord_netherlands.clm")
    assert grid_header.__class__.__name__ == "LPJmLMetaData"
    assert get_headersize(f"{test_path}/data/input/coord_netherlands.clm") == 43


def test_lpjmlinputtype(test_path):

    landuse = LPJmLInputType(6)

    assert landuse.name == "landuse"
    assert landuse.nband == 64
    assert landuse.type == float
    assert landuse.has_bands is True

    with_tillage = LPJmLInputType(7)

    assert with_tillage.name == "with_tillage"
    assert with_tillage.nband == 1
    assert with_tillage.type == int
    assert with_tillage.has_bands is False

    fertilizer_nr = LPJmLInputType(18)
    assert fertilizer_nr.name == "fertilizer_nr"
    assert fertilizer_nr.nband == 32
    assert fertilizer_nr.type == float
