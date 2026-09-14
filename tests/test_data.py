"""Test the LPJmLData class."""

import math
import numpy as np
import pytest

from pycoupler.data import (
    read_data,
    read_meta,
    read_header,
    get_headersize,
    LPJmLInputType,
    append_to_dict,
)
from tests.utils import ClmHeader, clm_file


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


@pytest.fixture
def clm_file_versions(request, sim_inputs):
    new_clm_file = sim_inputs / "test_clm.clm"
    with new_clm_file.open("wb") as f:
        header: ClmHeader = {
            "name": "LPJGRID",
            "version": request.param[0],
            "firstyear": 1900,
            "nyear": 1,
            "nbands": 1,
            "ncell": 10,
            "scalar": 4.2,
            "timestep": 7,
            "datatype": 3,
        }
        f.write(
            clm_file(header, big_endian=request.param[1], data=[0.01] * header["ncell"])
        )
    return new_clm_file


@pytest.mark.parametrize(
    ["clm_file_versions", "expected_version", "expected_endianness"],
    [((i, b), i, b) for b in [True, False] for i in range(1, 5)],
    indirect=["clm_file_versions"],
)
def test_read_header(request, clm_file_versions, expected_version, expected_endianness):
    header: dict[str, str | ClmHeader] = read_header(clm_file_versions, to_dict=True)

    assert header["name"] == "LPJGRID"
    assert header["header"]["version"] == expected_version
    assert header["header"]["firstyear"] == 1900
    assert header["header"]["nyear"] == 1
    assert header["header"]["nbands"] == 1
    assert header["header"]["ncell"] == 10
    assert header["endian"] == "big" if expected_endianness else "little"

    assert math.isclose(
        header["header"]["scalar"],
        (4.2 if expected_version >= 2 else 1.0),
        rel_tol=1e-4,
    )
    assert header["header"]["timestep"] == 7 if expected_version >= 4 else 1
    assert header["header"]["datatype"] == 3 if expected_version >= 3 else 1

    append_to_dict(header, {"test": "check"})
    assert header["test"] == "check"

    grid_header = read_header(clm_file_versions)
    assert grid_header.__class__.__name__ == "LPJmLMetaData"
    assert (
        get_headersize(clm_file_versions)
        == len(header["name"]) + 7 * 4 + (expected_version - 1) * 8
    )


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
