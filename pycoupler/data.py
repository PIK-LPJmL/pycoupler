import os
import struct
from enum import Enum
from collections.abc import Hashable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
from xarray.core.utils import either_dict_or_kwargs
from xarray.core.indexing import is_fancy_indexer
from xarray.core.indexes import isel_indexes

from pycoupler.utils import read_json


class LPJmLInputType(Enum):
    """Available Input types"""

    # input ids
    landuse: int = 6
    fertilizer_nr: int = 18
    manure_nr: int = 19
    residue_on_field: int = 8
    with_tillage: int = 7

    @property
    def nband(self):
        """get amount of bands"""
        if self.name == "landuse":
            return 64
        if self.name in ["fertilizer_nr", "manure_nr"]:
            return 32
        if self.name == "residue_on_field":
            return 16
        return 1

    @property
    def type(self):
        """get amount of bands"""
        if self.name in ["with_tillage"]:
            return int
        else:
            return float

    @property
    def bands(self):
        """check if multiple bands - better check for categorical bands
        (ADJUST WHEN REQUIRED)
        """
        return bool(self.nband > 1)


def append_to_dict(data_dict, data):
    """
    Append data along the third dimension to the data_dict.

    :param data_dict: Dictionary holding the data.
    :type data_dict: dict

    :param data: Dictionary with data.
                 Keys are ids/names, values are two-dimensional NumPy
                 arrays with dimensions (cells, bands).
    :type data: dict

    :return: Updated data_dict with the appended data.
    :rtype: dict
    """
    for key, value in data.items():
        if key in data_dict:
            data_dict[key] = np.dstack((data_dict[key], value))

        else:
            data_dict[key] = value

    return data_dict


class LPJmLData(xr.DataArray):
    """Class for LPJmL data"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLData, self).__init__(*args, **kwargs)

    def isel(
        self,
        indexers=None,
        drop=False,
        missing_dims="raise",
        **indexers_kwargs,
    ):
        """
        Return a new DataArray whose data is given by selecting indexes
        along the specified dimension(s).

        :param indexers: A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        :type indexers: dict, optional

        :param drop: If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        :type drop: bool, default: False

        :param missing_dims: What to do if dimensions that should be selected
            from are not present in the DataArray:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        :type missing_dims: {"raise", "warn", "ignore"}, default: "raise"

        :param indexers_kwargs: The keyword arguments form of ``indexers``.
        :type indexers_kwargs: {dim: indexer, ...}, optional

        :return: indexed
        :rtype: xarray.DataArray

        :seealso: Dataset.isel, DataArray.sel

        :Example:

        >>> da = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
        >>> da
        <xarray.DataArray (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
        Dimensions without coordinates: x, y

        >>> tgt_x = xr.DataArray(np.arange(0, 5), dims="points")
        >>> tgt_y = xr.DataArray(np.arange(0, 5), dims="points")
        >>> da = da.isel(x=tgt_x, y=tgt_y)
        >>> da
        <xarray.DataArray (points: 5)>
        array([ 0,  6, 12, 18, 24])
        Dimensions without coordinates: points
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            ds = self._to_temp_dataset()._isel_fancy(
                indexers, drop=drop, missing_dims=missing_dims
            )
            return self._from_temp_dataset(ds)

        # Much faster algorithm for when all indexers are ints, slices,
        # one-dimensional lists, or zero or one-dimensional np.ndarray's

        variable = self._variable.isel(indexers, missing_dims=missing_dims)
        indexes, index_variables = isel_indexes(
            self.xindexes,
            {f"{k} ({self.name})": v for k, v in indexers.items() if k == "band"},
        )

        coords = {}
        for coord_name, coord_value in self._coords.items():
            if coord_name in index_variables:
                coord_value = index_variables[coord_name]
            else:
                coord_indexers = {
                    k: v for k, v in indexers.items() if k in coord_value.dims
                }
                if coord_indexers:
                    coord_value = coord_value.isel(coord_indexers)
                    if drop and coord_value.ndim == 0:
                        continue
            coords[coord_name] = coord_value

        return self._replace(variable=variable, coords=coords, indexes=indexes)

    def add_meta(self, meta_data):
        """
        Add meta data to the data array.

        :param meta_data: Meta data to be added to the data array.
        :type meta_data: LPJmLMetaData
        """
        if isinstance(meta_data, LPJmLMetaData):
            self.attrs["standard_name"] = meta_data.variable
            self.attrs["long_name"] = meta_data.long_name
            self.attrs["units"] = meta_data.unit
            self.attrs["source"] = meta_data.source
            self.attrs["history"] = meta_data.history

            if meta_data.cellsize_lat != meta_data.cellsize_lon:
                raise ValueError(
                    "Cell sizes in latitude and longitude direction must be " "equal!"
                )
            else:
                self.attrs["cellsize"] = meta_data.cellsize_lon

            if hasattr(meta_data, "global_attrs"):
                self.attrs["institution"] = meta_data.global_attrs["institution"]
                self.attrs["contact"] = meta_data.global_attrs["contact"]
                self.attrs["comment"] = meta_data.global_attrs["comment"]
        else:
            raise TypeError("meta_data must be of type LPJmLMetaData!")

    def get_neighbourhood(self, id=True, cellsize=0.5):
        """
        Get the IDs of all neighbouring cells within a given size of cells.
        :param id: If True, return cell ids, else return cell indices
        :type id: bool
        :param cellsize: Size of cells in degrees.
        :type cellsize: float
        :return: Array with the IDs of all neighbouring cells.
        :rtype: numpy.ndarray
        """

        # Get the coordinates of all cells
        coords_np = np.array([self.cell.lat.values, self.cell.lon.values]).T

        # Build a KDTree for fast nearest-neighbour lookup
        tree = KDTree(coords_np)

        if "cellsize" in self.attrs:
            cellsize = self.cellsize  # in degrees

        # Find all neighbours within the given size of cells
        neighbour_indices = tree.query_ball_point(coords_np, r=cellsize * 1.5)

        # Initialize the array to hold the neighbour cell IDs
        max_neighbours = 8
        n_cells = len(self.cell)
        neighbour_ids = np.full((n_cells, max_neighbours), -9999, dtype=int)

        # Loop over all cells and find their neighbours
        cell_indices = self.cell.values
        for i in range(n_cells):
            # Get the indices of all neighbours for this cell
            current_neighbours = neighbour_indices[i]

            # Remove the current cell from the list of neighbours
            current_neighbours = [n for n in current_neighbours if n != i]

            # Truncate the list to at most max_neighbours
            current_neighbours = current_neighbours[:max_neighbours]

            # Store the neighbour cell IDs in the output array
            if len(current_neighbours) > 0 and id:
                neighbour_ids[i, : len(current_neighbours)] = cell_indices[
                    current_neighbours
                ]
            elif len(current_neighbours) > 0 and not id:
                neighbour_ids[i, : len(current_neighbours)] = current_neighbours

        neighbours = LPJmLData(
            data=neighbour_ids,
            dims=("cell", "neighbour"),
            coords=dict(
                cell=self.cell.values,
                lon=(["cell"], self.lon.values),
                lat=(["cell"], self.lat.values),
                neighbour=np.arange(8),
            ),
            name="neighbourhood",
        )

        return neighbours

    def transform(self):
        """TODO: implement function to convert cell into lon/lat format"""
        pass


class LPJmLDataSet(xr.Dataset):
    """Class for LPJmL data sets."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLDataSet, self).__init__(*args, **kwargs)

        if self.data_vars and ("cellsize" in self[list(self.data_vars)[0]].attrs):
            first_attrs = self[list(self.data_vars)[0]].attrs
            self.attrs["source"] = first_attrs["source"]
            self.attrs["history"] = first_attrs["history"]
            self.attrs["cellsize"] = first_attrs["cellsize"]

            if "institution" in first_attrs:
                self.attrs["institution"] = first_attrs["institution"]
                self.attrs["contact"] = first_attrs["contact"]
                self.attrs["comment"] = first_attrs["comment"]

    def to_numpy(self):
        """Return data as numpy array."""
        return {key: value.to_numpy() for key, value in self.data_vars.items()}

    def _construct_dataarray(self, name: Hashable) -> LPJmLData:
        """Construct a LPJmLData by indexing this dataset"""

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = xr.core.dataset._get_virtual_variable(
                self._variables, name, self.dims
            )

        needed_dims = set(variable.dims)

        coords: dict[Hashable, xr.core.variable.Variable] = {}
        # preserve ordering
        for k in self._variables:
            if k in self._coord_names and (set(self.variables[k].dims) <= needed_dims):
                coords[k] = self.variables[k]

        indexes = xr.core.indexes.filter_indexes_from_coords(self._indexes, set(coords))

        # TODO: this is a hack to get around the fact that we don't have
        #  a proper way to represent band dimensions in xarray

        # get the corresponding band dimension
        band_dim = [
            dim for dim in variable._dims if dim.startswith("band") and dim != "band"
        ]
        if band_dim:
            variable._dims = variable._parse_dimensions(
                [
                    dim if not dim.startswith("band") else "band"
                    for dim in variable._dims
                ]
            )

        # get the corresponding "band" index and delete all other band indexes
        band_idx = [key for key in coords if key.startswith("band") and key != "band"]
        if band_idx:
            for key in band_idx:
                if name not in key:
                    del indexes[key]
                else:
                    band_idx_name = key
            indexes["band"] = indexes.pop(band_idx_name)
            indexes["band"].index.name = "band"

        # get the corresponding "band" coords and delete all other band coords
        band_coords = [
            key for key in coords if key.startswith("band") and key != "band"
        ]
        if band_coords:
            for key in band_coords:
                if name not in key:
                    del coords[key]
                else:
                    band_coords_name = key
            coords["band"] = coords.pop(band_coords_name)
            coords["band"]._dims = ("band",)

        # rename the band dimension to only name "band"
        if name.startswith("band") and name != "band":
            name = "band"

        return LPJmLData(variable, coords, name=name, indexes=indexes, fastpath=True)


def read_data(file_name, var_name=None):
    """Read netcdf file and return data as numpy array or xarray.DataArray.
    :param file_name: path to netcdf file
    :type file_name: str
    :param var_name: name of variable to be read
    :type var_name: str
    :return: data as LPJmLData (xarray.DataArray)
    :rtype: LPJmLData
    """
    with xr.open_dataset(file_name, decode_times=False, mask_and_scale=False) as data:
        units, reference_date = data.time.attrs["units"].split("since")
        date_time = pd.date_range(
            start=reference_date, periods=data.sizes["time"], freq="MS"
        )
        data["time"] = date_time.year
        if var_name:
            data = data[var_name]
            data = LPJmLData(data)
        else:
            data = LPJmLDataSet(data)
        return data


class LPJmLMetaData:
    """LPJmL meta data class that can be easily accessed,
    converted to a dictionary or written as a json file.

    :param config_dict: takes a dictionary (ideally LPJmL config dictionary)
        and builds up a nested LpjmLConfig class with corresponding fields
    :type config_dict: dict
    """

    def __init__(self, meta_dict):
        """Constructor method"""
        self.__dict__.update(meta_dict)
        if "band_names" not in self.__dict__ and hasattr(self, "nbands"):
            self.band_names = [str(ii) for ii in range(self.nbands)]

    def to_dict(self):
        """Convert class object to dictionary"""

        def obj_to_dict(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            result = {}
            for key, val in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                element = []
                if isinstance(val, list):
                    for item in val:
                        element.append(obj_to_dict(item))
                else:
                    element = obj_to_dict(val)
                result[key] = element
            return result

        return obj_to_dict(self)

    def __repr__(self, sub_repr=False):
        """Representation of the LPJmL Meta object"""
        summary_attr = [
            "sim_name",
            "source",
            "history",
            "variable",
            "long_name",
            "unit",
            "nbands",
            "band_names",
            "nyear",
            "firstyear",
            "lastyear",
            "cellsize_lon",
            "cellsize_lat",
            "ncell",
            "firstcell",
            "datatype",
            "scalar",
        ]
        other_attr = [
            to_repr for to_repr in self.__dict__ if to_repr not in summary_attr
        ]
        if sub_repr:
            spacing = "\n  "
            summary = "Meta data:"
        else:
            summary = f"<pycoupler.{self.__class__.__name__}>"
            spacing = "\n"
        # TODO: add solution for sub dicts else error in repr
        summary = spacing.join(
            [
                summary,
                f"  * sim_name      {self.sim_name}",
                f"  * source        {self.source}",
                f"  * history       {self.history}",
                f"  * variable      {self.variable}",
                f"  * long_name     {self.long_name}",
                f"  * unit          {self.unit}",
                f"  * nbands        {self.nbands}",
                f"  * band_names    {self.band_names}",
                f"  * nyear         {self.nyear}",
                f"  * firstyear     {self.firstyear}",
                f"  * lastyear      {self.lastyear}",
                f"  * cellsize_lon  {self.cellsize_lon}",
                f"  * cellsize_lat  {self.cellsize_lat}",
                f"  * ncell         {self.ncell}",
                f"  * firstcell     {self.firstcell}",
                f"  * datatype      {self.datatype}",
                f"  * scalar        {self.scalar}",
            ]
        )
        if other_attr:
            summary_list = [summary]
            summary_list.extend(
                [
                    f"  * {torepr}{(13-len(torepr))*' '} {getattr(self, torepr)}"
                    for torepr in other_attr
                ]
            )
            summary = spacing.join(summary_list)
        return summary


def read_meta(file_name):
    """Read meta data from json file and return as LPJmLMetaData object.
    :param file_name: path to json file
    :type file_name: str
    :return: meta data as dictionary
    """
    return LPJmLMetaData(read_json(file_name))


def project_data(output_path, file_name):
    """Project tabular data to grid."""
    grid = read_data(f"{output_path}/grid.nc4").astype("float32")

    all_output = pd.read_csv(f"{output_path}/{file_name}")
    ts = pd.pivot_table(
        all_output, values="value", index=["year", "cell"], columns=["variable"]
    )

    # create xarray dataset
    ds = xr.Dataset()

    # create time dimension
    ds["time"] = ts.index.levels[0]

    # create cell dimension
    ds["cell"] = grid.cellid

    # create data arrays for each variable
    for var in ts.columns:
        da = xr.DataArray(
            ts[var].values.reshape((len(ds["time"]), len(ds["cell"]))),
            dims=("time", "cell"),
            coords={"time": ds["time"], "cell": ds["cell"]},
            name=var,
        )
        ds[var] = da

    return ds


# Function has been derived from the lpjmlkit R package
#   https://github.com/PIK-LPJmL/lpjmlkit
#   Author of original R function: Sebastian Ostberg
def read_header(filename, return_dict=False, force_version=None, verbose=False):
    """
    Read header (any version) from LPJmL input/output file

    Reads a header from an LPJmL clm file. CLM is the default format used for
    LPJmL input files and can also be used for output files.
    :param filename: Filename to read header from.
    :type filename: str
    :param return_dict: If True, return header as dictionary. If False, return
        as LPJmLMetaData object.
    :type return_dict: bool
    :param force_version: Manually set clm version. The default value `NULL`
        means that the version is determined automatically from the header. Set
        only if the version number in the file header is incorrect.
    :type force_version: int
    :param verbose: If `TRUE` (the default), `read_header` provides some
        feedback when using default values for missing parameters. If `FALSE`,
        only errors are reported.
    :type verbose: bool
    :return: A LPJmLMetaData object or a dictionary with the header
        information.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    with open(filename, "rb") as f:
        # Read the first 30 bytes to determine the header name
        headername_raw = f.read(30)
        headername = "".join(chr(b) for b in headername_raw)
        first_non_alnum_index = next(
            (
                i
                for i, char in enumerate(headername)
                if not (char.isalnum() or char == "_")
            ),
            len(headername),  # noqa
        )
        headername = headername[:first_non_alnum_index]

        for i, char in enumerate(headername):
            if not (char.isalnum() or char == "_"):
                headername = headername[:i]
                break

        if not headername.startswith("LPJ"):
            raise ValueError(f"Invalid header name {headername}")
        if headername == "LPJRESTART":
            raise ValueError(
                "LPJRESTART header detected. This function does not support"
                + " restart headers at the moment."
            )

        # Skip over the header
        f.seek(len(headername))

        # Determine file endian. Try platform-specific endian as default.
        endian = "little"
        version = struct.unpack(f"<i", f.read(4))[0]

        if version & 0xFF == 0:
            endian = "big"
            f.seek(len(headername))
            version = struct.unpack(f">i", f.read(4))[0]

        if force_version is not None and force_version != version:
            if verbose:
                print(f"Forcing header version to {force_version}")
            version = force_version

        # Read main header attributes that are included in all header versions
        if endian == "little":
            headerdata = struct.unpack("<6i", f.read(24))
        else:
            headerdata = struct.unpack(">6i", f.read(24))

        headerdata_dict = {
            "order": headerdata[0],
            "firstyear": headerdata[1],
            "nyear": headerdata[2],
            "firstcell": headerdata[3],
            "ncell": headerdata[4],
            "nbands": headerdata[5],
        }

        if version == 2:
            if endian == "little":
                extra_data = struct.unpack("<2f", f.read(8))
            else:
                extra_data = struct.unpack(">2f", f.read(8))
            headerdata_dict.update(
                {"cellsize_lon": extra_data[0], "scalar": extra_data[1]}
            )

        if version >= 3:
            if endian == "little":
                extra_data = struct.unpack("<3f", f.read(12))
                datatype = struct.unpack("<i", f.read(4))[0]
            else:
                extra_data = struct.unpack(">3f", f.read(12))
                datatype = struct.unpack(">i", f.read(4))[0]
            headerdata_dict.update(
                {
                    "cellsize_lon": extra_data[0],
                    "scalar": extra_data[1],
                    "cellsize_lat": extra_data[2],
                    "datatype": datatype,
                }
            )

        if version == 4:
            if endian == "little":
                nstep = struct.unpack("<i", f.read(4))[0]
                timestep = struct.unpack("<i", f.read(4))[0]
            else:
                nstep = struct.unpack(">i", f.read(4))[0]
                timestep = struct.unpack(">i", f.read(4))[0]
            headerdata_dict.update({"nstep": nstep, "timestep": timestep})
        else:
            if len(headerdata_dict) == 6:
                headerdata_dict.update(
                    {
                        "cellsize_lon": 0.5,
                        "scalar": 1,
                        "cellsize_lat": 0.5,
                        "datatype": 1,
                        "nstep": 1,
                        "timestep": 1,
                    }
                )
                if verbose:
                    print(
                        "Note: Type 1 header. Adding default values for"
                        + " cellsize, scalar, datatype, nstep and timestep"
                        + " which may not be correct in all cases."
                    )
            if len(headerdata_dict) == 8:
                headerdata_dict.update(
                    {
                        "cellsize_lat": headerdata_dict["cellsize_lon"],
                        "datatype": 1,
                        "nstep": 1,
                        "timestep": 1,
                    }
                )
                if verbose:
                    print(
                        "Note: Type 2 header. Adding default values for"
                        + " datatype, nstep and timestep which may not be"
                        + " correct in all cases."
                    )
            if len(headerdata_dict) == 10:
                headerdata_dict.update({"nstep": 1, "timestep": 1})
                if verbose:
                    print(
                        "Note: Type 3 header. Adding default values for"
                        + " nstep and timestep which may not be correct in all"
                        + " cases."
                    )

        if verbose and headerdata_dict.get("datatype") is None:
            print(
                f"Warning: Invalid datatype {headerdata_dict['datatype']} in"
                + f" header read from {filename}"
            )

    if return_dict:
        return {
            "name": headername,
            "header": {"version": version, **headerdata_dict},
            "endian": endian,
        }
    else:
        return LPJmLMetaData(
            {
                "sim_name": None,
                "source": None,
                "history": None,
                "variable": headername,
                "long_name": None,
                "unit": None,
                "nbands": headerdata_dict["nbands"],
                "band_names": None,
                "nyear": headerdata_dict["nyear"],
                "firstyear": headerdata_dict["firstyear"],
                "lastyear": headerdata_dict["firstyear"]
                + headerdata_dict["nyear"]
                - 1,  # noqa
                "cellsize_lon": headerdata_dict["cellsize_lon"],
                "cellsize_lat": headerdata_dict.get(
                    "cellsize_lat", headerdata_dict["cellsize_lon"]
                ),  # noqa
                "ncell": headerdata_dict["ncell"],
                "firstcell": headerdata_dict["firstcell"],
                "nstep": headerdata_dict["nstep"],
                "timestep": headerdata_dict["timestep"],
                "datatype": headerdata_dict["datatype"],
                "scalar": headerdata_dict["scalar"],
            }
        )


# Function has been derived from the lpjmlkit R package
#   https://github.com/PIK-LPJmL/lpjmlkit
#   Author of original R function: Sebastian Ostberg
def get_headersize(filename):
    """
    Get the size of the header in an LPJmL input/output file.
    :param filename: Filename to read header size from.
    :type filename: str
    :return: Size of the header in bytes.
    :rtype: int
    """
    header = read_header(filename, return_dict=True)
    version = header["header"]["version"]
    if version < 1 or version > 4:
        raise ValueError("Invalid header version. Expecting value between 1 and 4.")

    headersize = len(header["name"]) + {1: 7, 2: 9, 3: 11, 4: 13}[version] * 4
    return headersize
