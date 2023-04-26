import os
import tempfile
import xarray as xr
import numpy as np

from enum import Enum
from subprocess import run


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
        elif self.name == "with_tillage":
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
    # collect via key value pairs
    return_dict = {}
    # utility function to get general temp folder for every system
    temp_dir = tempfile.gettempdir()

    if not start_year and not end_year:
        start_year = end_year = coupler.config.start_coupling - 1

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
        if start_year and not end_year:
            # run cut clm file before start year
            run(cut_clm_start, stdout=open(os.devnull, 'wb'))
            use_tmp = '1'
        elif not start_year and end_year:
            # run cut clm file after end year
            run([f"{model_path}/bin/cutclm",
                 "-end",
                 str(end_year),
                 sock_inputs[key]['name'],
                 f"{temp_dir}/2_{file_name_tmp}"],
                stdout=open(os.devnull, 'wb'))
            use_tmp = '2'
        else:
            # run cut clm file before start year and after end year in
            #   sequence
            run(cut_clm_start, stdout=open(os.devnull, 'wb'))
            # cannot deal with overwriting a temp file with same name
            cut_clm_end = [f"{model_path}/bin/cutclm",
                           "-end", str(end_year),
                           f"{temp_dir}/1_{file_name_tmp}",
                           f"{temp_dir}/2_{file_name_tmp}"]
            run(cut_clm_end, stdout=open(os.devnull, 'wb'))
            use_tmp = '2'
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
            grid_file, f"{temp_dir}/{use_tmp}_{file_name_tmp}",
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


def read_netcdf(file_name, var_name=None, return_xarray=True):
    """Read netcdf file and return data as numpy array or xarray.DataArray.
    :param file_name: path to netcdf file
    :type file_name: str
    :param var_name: name of variable to be read
    :type var_name: str
    :param return_xarray: return data as xarray.DataArray (True) or NumPy array
        (False)
    :type return_xarray: bool
    :return: data as numpy array or xarray.DataArray
    :rtype: numpy.ndarray or xarray.DataArray
    """
    data = xr.open_dataset(file_name,
                           decode_times=True,
                           mask_and_scale=False)
    if var_name:
        data = data[var_name]
        if not return_xarray:
            data = data.to_numpy()
    return data


def read_coupled_input(coupler, sim_path, return_xarray=False):
    """Read coupled input data from netcdf files """
    # read coupled input data from netcdf files (as xarray.DataArray)
    inputs = {key: read_netcdf(
        f"{sim_path}/input/{key}.nc",
        var_name=key
    ) for key in coupler.config.get_input_sockets(id_only=True)}

    # define longitide and latitude DataArray (workaround to reduce dims to
    #   cells)
    lons = xr.DataArray(coupler.grid[:, 0], dims="cell")
    lats = xr.DataArray(coupler.grid[:, 1], dims="cell")

    # create same format as before but with selected numpy arrays instead of
    #   xarray.DataArray
    input_data = {key: inputs[key].sel(
        longitude=lons, latitude=lats,
        time=coupler.config.start_coupling - 1, method="nearest"
    ).transpose("cell", ...) for key in inputs}

    # return input data as xarray.DataArray if requested
    if not return_xarray:
        input_data = {key: input_data[key].to_numpy() for key in input_data}

    return input_data


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
