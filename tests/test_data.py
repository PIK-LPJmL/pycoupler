"""Test the LPJmLData class."""

from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from pycoupler.data import (
    LPJmLData,
    LPJmLDataSet,
    append_to_dict,
    get_headersize,
    LPJmLInputType,
    read_data,
    read_header,
    read_meta,
)


def _sample_lpjml_data():
    """Create a tiny LPJmLData object with cell+lat/lon information."""
    cell_ids = np.array([100, 101, 200, 201])
    lat = np.array([50.0, 50.0, 49.5, 49.5])
    lon = np.array([-1.0, -0.5, -1.0, -0.5])
    time = np.array([2000, 2001])

    values = np.arange(cell_ids.size * time.size, dtype=float).reshape(
        cell_ids.size, time.size
    )
    return LPJmLData(
        data=values,
        dims=("cell", "time"),
        coords=dict(
            cell=cell_ids,
            time=time,
            lat=("cell", lat),
            lon=("cell", lon),
        ),
        name="soilc",
    )


def test_lpjmldata_transform_roundtrip():
    """Validate cell <-> lon/lat transforms match lpjmlkit behaviour."""
    data = _sample_lpjml_data()

    lon_lat = data.transform("lon_lat")
    assert set(lon_lat.dims) == {"lat", "lon", "time"}
    np.testing.assert_allclose(lon_lat.lat.values, np.array([50.0, 49.5]))
    np.testing.assert_allclose(lon_lat.lon.values, np.array([-1.0, -0.5]))

    roundtrip = lon_lat.transform("cell")
    assert set(roundtrip.dims) == {"cell", "time"}

    roundtrip_sorted = roundtrip.sortby("lon").sortby("lat", ascending=False)
    expected = data.sortby("lon").sortby("lat", ascending=False)

    roundtrip_sorted = roundtrip_sorted.assign_coords(
        cell=("cell", np.arange(roundtrip_sorted.sizes["cell"]))
    )
    expected = expected.assign_coords(cell=("cell", np.arange(expected.sizes["cell"])))

    xr.testing.assert_allclose(roundtrip_sorted, expected)


def test_lpjmldataset_transform_and_netcdf(tmp_path):
    """Ensure dataset transform enables writing gridded NetCDF output."""
    soilc = _sample_lpjml_data()
    ds = LPJmLDataSet({"soilc": soilc})

    lon_lat_ds = ds.transform("lon_lat")
    lon_lat_var = lon_lat_ds["soilc"]
    assert {"lat", "lon"} <= set(lon_lat_var.dims)

    nc_path = tmp_path / "soilc.nc4"
    lon_lat_var.to_netcdf(nc_path)
    with xr.open_dataset(nc_path) as reopened:
        reopened_var = reopened["soilc"].transpose(*lon_lat_var.dims)
        np.testing.assert_allclose(reopened_var.values, lon_lat_var.values)

    cell_ds = lon_lat_ds.transform("cell")
    cell_sorted = cell_ds["soilc"].sortby("lon").sortby("lat", ascending=False)
    expected = soilc.sortby("lon").sortby("lat", ascending=False)
    cell_sorted = cell_sorted.assign_coords(
        cell=("cell", np.arange(cell_sorted.sizes["cell"]))
    )
    expected = expected.assign_coords(cell=("cell", np.arange(expected.sizes["cell"])))
    xr.testing.assert_allclose(cell_sorted, expected)


def test_write_lpjmldata_netcdf_helper(tmp_path):
    """Ensure helper writes grid and non-grid variables."""
    soilc = _sample_lpjml_data()
    target = tmp_path / "soilc.nc4"
    soilc.to_netcdf(target)
    assert target.exists()
    with xr.open_dataset(target) as reopened:
        assert reopened["soilc"].dims == soilc.transform("lon_lat").dims
        np.testing.assert_allclose(
            reopened["soilc"].values,
            soilc.transform("lon_lat").values,
        )

    world = LPJmLData(
        data=np.array([1.0, 2.0]),
        dims=("time",),
        coords={"time": [2000, 2001]},
        name="world_var",
    )
    world_target = tmp_path / "world_var.nc4"
    world.to_netcdf(world_target)
    with xr.open_dataset(world_target) as reopened:
        np.testing.assert_allclose(reopened["world_var"].values, world.values)


def test_lpjmldata_method_to_netcdf(tmp_path):
    soilc = _sample_lpjml_data()
    target = tmp_path / "method_soilc.nc4"
    result_path = soilc.to_netcdf(target)
    assert Path(result_path).exists()


def test_lpjmldata_global_attrs_passthrough(tmp_path):
    soilc = _sample_lpjml_data()
    soilc.attrs["_global_attrs"] = {"title": "Test Title", "institution": "PIK"}
    target = tmp_path / "global.nc4"
    soilc.to_netcdf(target)
    with xr.open_dataset(target) as reopened:
        assert reopened.attrs["title"] == "Test Title"
        assert reopened.attrs["institution"] == "PIK"


def test_netcdf_fill_values_are_finite(tmp_path):
    soilc = _sample_lpjml_data()
    target = tmp_path / "finite_fill.nc4"
    soilc.to_netcdf(target)

    with Dataset(target) as nc:
        soilc_var = nc.variables["soilc"]
        assert abs(float(soilc_var._FillValue)) < 1000
        for coord_name in ("time", "lat", "lon"):
            coord_var = nc.variables[coord_name]
            assert "_FillValue" not in coord_var.ncattrs()


def test_lpjmldataset_to_netcdf_separate(tmp_path):
    soilc = _sample_lpjml_data()
    world = LPJmLData(
        data=np.array([1.0, 2.0]),
        dims=("time",),
        coords={"time": [2000, 2001]},
        name="world_var",
    )
    ds = LPJmLDataSet({"soilc": soilc, "world_var": world})

    out_dir = tmp_path / "nc_out"
    files = ds.to_netcdf(out_dir, file_prefix="run")
    assert set(files) == {"soilc", "world_var"}
    for file_path in files.values():
        assert Path(file_path).exists()


def test_lpjmldataset_to_netcdf_combined(tmp_path):
    soilc = _sample_lpjml_data()
    world = LPJmLData(
        data=np.array([1.0, 2.0]),
        dims=("time",),
        coords={"time": [2000, 2001]},
        name="world_var",
    )
    ds = LPJmLDataSet({"soilc": soilc, "world_var": world})

    target = tmp_path / "combined.nc4"
    result = ds.to_netcdf(target, per_variable=False)
    assert Path(result).exists()
    with xr.open_dataset(result) as reopened:
        assert {"lat", "lon"} <= set(reopened["soilc"].dims)
        np.testing.assert_allclose(
            reopened["soilc"].values,
            soilc.transform("lon_lat").values,
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
