import os
import tempfile
import json
from typing import Any
from collections.abc import Hashable, Mapping
from enum import Enum

import numpy as np
import pandas as pd

from subprocess import run, CalledProcessError

import xarray as xr
from scipy.spatial import KDTree


class Inputs(Enum):
    """Available Inputs"""
    landuse: int = 6  # number of bands in landuse data
    fertilizer_nr: int = 18  # number of bands in fertilizer data
    manure_nr: int = 19  # number of bands in manure data
    residue_on_field: int = 8  # number of bands in residue data
    with_tillage: int = 7  # number of bands in tillage data

    @property
    def nband(self):
        """ get amount of bands
        """
        if self.name == "landuse":
            return 64
        elif self.name in ["fertilizer_nr", "manure_nr", "residue_on_field"]:
            return 32
        else:  # "with_tillage"
            return 1

    @property
    def type(self):
        """ get amount of bands
        """
        if self.name in ["with_tillage"]:
            return int
        else:
            return float

    @property
    def bands(self):
        """ check if multiple bands - better check for categorical bands
        (ADJUST WHEN REQUIRED)
        """
        if self.nband > 1:
            return True
        else:
            return False


class LpjmlTypes(Enum):
    """Available datatypes
    """
    LPJ_BYTE: int = 0
    LPJ_SHORT: int = 1
    LPJ_INT: int = 2
    LPJ_FLOAT: int = 3
    LPJ_DOUBLE: int = 4

    @property
    def type(self):
        """Convert LPJmL data type to Python data types
        """
        if self.value > 2:
            return float
        else:
            return int


def copy_coupled_input(coupler,
                       sim_path,
                       model_path,
                       start_year=None,
                       end_year=None):
    """Copy and convert and save input files as NetCDF4 files to input
    directory for selected years to make them easily readable as well as to
    avoid large file sizes.
    :param coupler: LPJmLCoupler object
    :type coupler: LPJmLCoupler
    :param sim_path: path to simulation directory
    :type sim_path: str
    :param model_path: path to model directory
    :type model_path: str
    :param start_year: start year of input data
    :type start_year: int
    :param end_year: end year of input data
    :type end_year: int
    """
    input_path = f"{sim_path}/input"
    # get defined input sockets
    if not os.path.isdir(input_path):
        os.makedirs(input_path)
        print(f"Created input path '{input_path}'")
    sock_inputs = coupler.config.get_input_sockets()

    # utility function to get general temp folder for every system
    temp_dir = tempfile.gettempdir()

    if not start_year:
        start_year = coupler.config.start_coupling - 1
    if not end_year:
        end_year = coupler.config.start_coupling - 1

    # iterate over each inputs to be send via sockets (get initial values)
    for key in sock_inputs:
        # check if working on the cluster (workaround by Ciaron)
        #   (might be adjusted to the new cluster coming soon ...)
        if coupler.config.inpath and (
            not sock_inputs[key]['name'].startswith("/")
        ):
            sock_inputs[key]['name'] = (
                f"{coupler.config.inpath}/{sock_inputs[key]['name']}"
            )
        # get input file name
        file_name_clm = sock_inputs[key]['name'].split("/")[-1]
        # name tmp file after original name (even though could be random)
        file_name_tmp = (
            f"{file_name_clm.split('.')[0]}_tmp.clm"
        )
        # predefine cut clm command for reusage
        cut_clm_start = [f"{model_path}/bin/cutclm",
                         str(start_year),
                         sock_inputs[key]['name'],
                         f"{temp_dir}/1_{file_name_tmp}"]

        start_year_check = start_year
        # run cut clm file before start year and after end year in sequence
        while True:
            try:
                run(cut_clm_start, stdout=open(os.devnull, 'wb'), check=True)
                break
            except CalledProcessError:
                start_year_check += 1
                if start_year_check > end_year:
                    raise ValueError(
                        f"Could not find input file for '{key}' "
                        f"between {start_year} and {end_year}!"
                    )
                cut_clm_start[1] = str(start_year_check)

        # cannot deal with overwriting a temp file with same name
        cut_clm_end = [f"{model_path}/bin/cutclm",
                       "-end", str(end_year),
                       f"{temp_dir}/1_{file_name_tmp}",
                       f"{temp_dir}/2_{file_name_tmp}"]
        run(cut_clm_end, stdout=open(os.devnull, 'wb'))
        # a flag for multi (categorical) band input - if true, set
        #   "-landuse"
        if getattr(Inputs, key).bands:
            is_multiband = "-landuse"
        else:
            is_multiband = None
        # a flag for integer input - if true, set "-int"
        if getattr(Inputs, key).type == int:
            is_int = "-intnetcdf"
        else:
            is_int = None
        # default grid file (only valid for 0.5 degree inputs)
        if coupler.config.input.coord.name.startswith("/"):
            grid_file = coupler.config.input.coord.name
        else:
            grid_file = (
                f"{coupler.config.inpath}/{coupler.config.input.coord.name}"
            )
        # convert clm input to netcdf files
        conversion_cmd = [
            f"{model_path}/bin/clm2cdf", is_int, is_multiband, key,
            grid_file, f"{temp_dir}/2_{file_name_tmp}",
            f"{input_path}/{key}.nc"
        ]
        if None in conversion_cmd:
            conversion_cmd.remove(None)
        run(conversion_cmd)
        # remove the temporary clm (binary) files, 1_* is not created in
        #   every case
        if os.path.isfile(f"{temp_dir}/1_{file_name_tmp}"):
            os.remove(f"{temp_dir}/1_{file_name_tmp}")
        if os.path.isfile(f"{temp_dir}/2_{file_name_tmp}"):
            os.remove(f"{temp_dir}/2_{file_name_tmp}")


def read_coupled_input(coupler,
                       sim_path,
                       start_year=None,
                       end_year=None):
    """Read coupled input data from netcdf files """
    # read coupled input data from netcdf files (as xarray.DataArray)
    inputs = {key: read_data(
        f"{sim_path}/input/{key}.nc",
        var_name=key
    ) for key in coupler.config.get_input_sockets(id_only=True)}

    inputs = LPJmLDataSet(inputs)
    # define longitide and latitude DataArray (workaround to reduce dims to
    #   cells)
    lons = xr.DataArray(coupler.grid[:, 0], dims="cell")
    lats = xr.DataArray(coupler.grid[:, 1], dims="cell")

    other_dim = [
        dim for dim in inputs.dims
        if dim not in ["time", "longitude", "latitude"]
    ]
    if other_dim:
        inputs = inputs.rename_dims({other_dim[0]: 'band'})

    if start_year and end_year:
        kwargs = {"time": [year for year in range(start_year, end_year + 1)]}
    elif start_year and not end_year:
        kwargs = {"time": [year for year in range(start_year,
                                                  max(inputs.time.values)+1)]}
    elif not start_year and end_year:
        kwargs = {"time": [year for year in range(min(inputs.time.values),
                                                  end_year + 1)]}
    else:
        kwargs = {}

    inputs.coords['time'] = pd.date_range(
        start=str(min(inputs.coords["time"].values)),
        end=str(max(inputs.coords["time"].values)+1),
        freq='A'
    )
    # create same format as before but with selected numpy arrays instead of
    # xarray.DataArray
    inputs = inputs.sel(
        longitude=lons, latitude=lats, method="nearest", **kwargs
    ).transpose("cell", ..., "time")

    return inputs


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
    """ Class for LPJmL data """
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLData, self).__init__(*args, **kwargs)

    def add_meta(self, meta_data):

        if isinstance(meta_data, LPJmLMetaData):

            self.attrs['standard_name'] = meta_data.variable
            self.attrs['long_name'] = meta_data.descr
            self.attrs['units'] = meta_data.unit
            self.attrs['source'] = meta_data.source
            self.attrs['history'] = meta_data.history

            if meta_data.cellsize_lat != meta_data.cellsize_lon:
                raise ValueError(
                    "Cell sizes in latitude and longitude direction must be "
                    "equal!"
                )
            else:
                self.attrs['cellsize'] = meta_data.cellsize_lon

            if hasattr(meta_data, 'global_attrs'):
                self.attrs['institution'] = meta_data.global_attrs.institution
                self.attrs['contact'] = meta_data.global_attrs.contact
                self.attrs['comment'] = meta_data.global_attrs.comment
        else:
            raise TypeError(
                "meta_data must be of type LPJmLMetaData!"
            )

    def get_neighbours(self, cellsize=0.5):
        """
        Get the IDs of all neighboring cells within a given size of cells.
        :param cellsize: Size of cells in degrees.
        :type cellsize: float
        :return: Array with the IDs of all neighboring cells.
        :rtype: numpy.ndarray
        """

        # Get the coordinates of all cells
        coords_np = np.array([self.cell.latitude.values,
                              self.cell.longitude.values]).T

        # Build a KDTree for fast nearest-neighbor lookup
        tree = KDTree(coords_np)

        if "cellsize" in self.attrs:
            cellsize = self.cellsize  # in degrees

        # Find all neighbors within the given size of cells
        neighbor_indices = tree.query_ball_point(coords_np, r=cellsize)

        # Initialize the array to hold the neighbor cell IDs
        max_neighbors = 8
        n_cells = len(self.cell)
        neighbor_ids = np.full((n_cells, max_neighbors), np.nan, dtype=int)

        # Loop over all cells and find their neighbors
        cell_indices = self.cell.values
        for i in range(n_cells):
            # Get the indices of all neighbors for this cell
            current_neighbors = neighbor_indices[i]

            # Remove the current cell from the list of neighbors
            current_neighbors = [n for n in current_neighbors if n != i]

            # Truncate the list to at most max_neighbors
            current_neighbors = current_neighbors[:max_neighbors]

            # Store the neighbor cell IDs in the output array
            if len(current_neighbors) > 0:
                neighbor_ids[i, :len(current_neighbors)] = cell_indices[
                    current_neighbors
                ]

        # Replace all NaNs with -9999
        neighbor_ids[neighbor_ids < 0] = -9999

        return neighbor_ids

    def transform(self):
        pass


class LPJmLDataSet(xr.Dataset):
    """Class for LPJmL data sets."""
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLDataSet, self).__init__(*args, **kwargs)

        if self.data_vars and (
            "cellsize" in self[list(self.data_vars)[0]].attrs
        ):
            first_attrs = self[list(self.data_vars)[0]].attrs
            self.attrs['source'] = first_attrs["source"]
            self.attrs['history'] = first_attrs["history"]
            self.attrs['cellsize'] = first_attrs["cellsize"]

            if "institution" in first_attrs:
                self.attrs['institution'] = first_attrs["institution"]
                self.attrs['contact'] = first_attrs["contact"]
                self.attrs['comment'] = first_attrs["comment"]

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
            if k in self._coord_names and (
                set(self.variables[k].dims) <= needed_dims
            ):
                coords[k] = self.variables[k]

        indexes = xr.core.indexes.filter_indexes_from_coords(
            self._indexes, set(coords)
        )

        # TODO: this is a hack to get around the fact that we don't have
        #  a proper way to represent band dimensions in xarray

        # get the corresponding band dimension
        band_dim = [
            dim for dim in variable._dims
            if dim.startswith('band') and dim != 'band'
        ]
        if band_dim:
            variable._dims = variable._parse_dimensions([
                dim if not dim.startswith('band') else 'band'
                for dim in variable._dims
            ])

        # get the corresponding "band" index and delete all other band indexes
        band_idx = [
            key for key in coords.keys()
            if key.startswith("band") and key != "band"
        ]
        if band_idx:
            for key in band_idx:
                if name not in key:
                    del indexes[key]
                else:
                    band_idx_name = key
            indexes['band'] = indexes.pop(band_idx_name)
            indexes['band'].index.name = "band"

        # get the corresponding "band" coords and delete all other band coords
        band_coords = [
            key for key in coords
            if key.startswith("band") and key != "band"
        ]
        if band_coords:
            for key in band_coords:
                if name not in key:
                    del coords[key]
                else:
                    band_coords_name = key
            coords['band'] = coords.pop(band_coords_name)
            coords['band']._dims = ('band',)

        # rename the band dimension to only name "band"
        if name.startswith("band") and name != "band":
            name = "band"

        return LPJmLData(
            variable, coords, name=name, indexes=indexes, fastpath=True
        )


def read_data(file_name, var_name=None):
    """Read netcdf file and return data as numpy array or xarray.DataArray.
    :param file_name: path to netcdf file
    :type file_name: str
    :param var_name: name of variable to be read
    :type var_name: str
    :param to_numpy: return data as numpy.ndarray (False)
    :type to_xarray: bool
    :return: data as numpy array or xarray.DataArray
    :rtype: numpy.ndarray or xarray.DataArray
    """
    data = xr.open_dataset(file_name,
                           decode_times=True,
                           mask_and_scale=False)

    if var_name:
        data = data[var_name]
        data = LPJmLData(data)

    return data


class LPJmLMetaData:
    """LPJmL meta data class that can be easily accessed,
    converted to a dictionary or written as a json file.

    :param config_dict: takes a dictionary (ideally LPJmL config dictionary)
        and builds up a nested LpjmLConfig class with corresponding fields
    :type config_dict: dict
    """
    def __init__(self, meta_dict):
        """Constructor method
        """
        self.__dict__.update(meta_dict)
        if "band_names" not in self.__dict__:
            self.band_names = [str(ii) for ii in range(self.nbands)]

    def to_dict(self):
        """Convert class object to dictionary
        """
        def obj_to_dict(obj):
            if not hasattr(obj, '__dict__'):
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
        """Representation of the LPJmL Meta object
        """
        summary_attr = ["sim_name", "source", "history", "variable", "descr",
                        "unit", "nbands", "band_names", "nyear", "firstyear",
                        "lastyear", "cellsize_lon", "cellsize_lat", "ncell",
                        "firstcell", "datatype", "scalar"]
        other_attr = [
            to_repr for to_repr in self.__dict__.keys()
            if to_repr not in summary_attr
        ]
        if sub_repr:
            spacing = "\n  "
            summary = "Meta data:"
        else:
            summary = f"<pycoupler.{self.__class__.__name__}>"
            spacing = "\n"
        summary = spacing.join([
            summary,
            f"  * sim_name      {self.sim_name}",
            f"  * source        {self.source}",
            f"  * history       {self.history}",
            f"  * variable      {self.variable}",
            f"  * descr         {self.descr}",
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

        ])
        if other_attr:
            summary_list = [summary]
            summary_list.extend([
                f"  * {torepr}{(13-len(torepr))*' '} {getattr(self, torepr)}"
                for torepr in other_attr
            ])
            summary = spacing.join(summary_list)
        return summary


def read_meta(file_name, ):
    """Read meta data from json file and return as dictionary.
    :param file_name: path to json file
    :type file_name: str
    :return: meta data as dictionary
    """
    with open(file_name, 'r') as f:
        return json.load(f, object_hook=LPJmLMetaData)
