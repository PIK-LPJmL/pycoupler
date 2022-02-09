import os
import tempfile
import xarray as xr
from enum import Enum
from subprocess import run

from pycoupler.config import read_config


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


def supply_inputs(config_file, historic_config_file, input_path, model_path,
                  start_year=None, end_year=None, return_xarray=True,
                  overwrite=False):
    """Convert and save input files as NetCDF4 files to input directory for
    selected years to make them easily readable as well as to avoid large
    file sizes
    :param config_file: file name including path if not current to config_file
        of coupled config (to get set socket inputs)
    :type config_file: str
    :param historic_config_file: file name including path if not current to
        config_file of historic config (to get original input paths)
    :type config_file: str
    :param input_path: path where the created nc file inputs should be stored
    :type input_path: str
    :param model_path: path to `LPJmL_internal` (lpjml repository)
    :type input_path: str
    :param start_year: first year of the created input file (years before are
        cut off). Defaults to None (no cutoff).
    :type start_year: int
    :param end_year: last year of the created input file (years after are
        cut off). Defaults to None (no cutoff).
    :type end_year: int
    :param return_xarray: if True (default) xarray.DataArray(s) are returned as
        dictionary values
    :type return_xarray: bool
    :param overwrite: if False (default) already existing input nc files in
        input_path are used if input type and start_year + end_year are
        matching.
    :type overwrite: bool
    :return: inputs dictionary with input keys (as keys) and
        xarray.DataArray(s) as value - CANNOT BE SENT TO LPJML (via Coupler) -
        please use `preprocess_inputs` first
    :rtype: dict (values as DataArray.array)
    """
    # get defined input sockets
    config = read_config(config_file)
    sock_inputs_keys = config.get_input_sockets().keys()
    # get input paths for historic run
    historic_config = read_config(file_name=historic_config_file)
    sock_inputs = historic_config.get_inputs(id_only=False,
                                             inputs=sock_inputs_keys)
    # collect via key value pairs
    return_dict = {}
    # utility function to get general temp folder for every system
    temp_dir = tempfile.gettempdir()
    # iterate over each inputs to be send via sockets (get initial values)
    for key in sock_inputs:
        # check if working on the cluster (workaround by Ciaron)
        #   (might be adjusted to the new cluster coming soon ...)
        if os.path.isdir('/p/system'):
            # only then the following is valid and has to be replaced by
            #   absolute path names
            if not sock_inputs[key]['name'].startswith('/p/'):
                sock_inputs[key]['name'] = (
                    "/p/projects/lpjml/input/" +
                    f"historical/{sock_inputs[key]['name']}"
                )
        # create file name with extension to inform about the extracted time
        #   span
        file_name_nc = (
            f"{key}_{start_year}-{end_year}.nc"
        )
        if overwrite or not os.path.isfile(f"{input_path}/{file_name_nc}"):
            # get input file name
            file_name_clm = sock_inputs[key]['name'].split("/")[-1]
            # name tmp file after original name (even though could be random)
            file_name_tmp = (
                f"{file_name_clm.split('.')[0]}_tmp.clm"
            )
            # predefine cut clm command for reusage
            cut_clm_start = [f"{model_path}/bin/cutclm",
                             str(start_year), sock_inputs[key]['name'],
                             f"{temp_dir}/1_{file_name_tmp}"]
            if start_year and not end_year:
                # run cut clm file before start year
                run(cut_clm_start, stdout=open(os.devnull, 'wb'))
                use_tmp = '1'
            elif not start_year and end_year:
                # run cut clm file after end year
                run([f"{model_path}/bin/cutclm",
                     "-end", str(end_year), sock_inputs[key]['name'],
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
                xarg = "-landuse"
            else:
                xarg = ""
            # default grid file (only valid for 0.5 degree inputs)
            grid_file = (
                "/p/projects/lpjml/input/historical/input_VERSION2/grid.bin"
            )
            # convert clm input to netcdf files
            run([f"{model_path}/bin/clm2cdf", xarg, key, grid_file,
                 f"{temp_dir}/{use_tmp}_{file_name_tmp}",
                 f"{input_path}/{file_name_nc}"])
            # remove the temporary clm (binary) files, 1_* is not created in
            #   every case
            if os.path.isfile(f"{temp_dir}/1_{file_name_tmp}"):
                os.remove(f"{temp_dir}/1_{file_name_tmp}")
            if os.path.isfile(f"{temp_dir}/2_{file_name_tmp}"):
                os.remove(f"{temp_dir}/2_{file_name_tmp}")
            # collect created input filenames and connect with input key
        return_dict[key] = file_name_nc
    if return_xarray:
        return {key: getattr(xr.open_dataset(
            f"{input_path}/{return_dict[key]}",
            decode_times=True
        ), key) for key in return_dict}
    else:
        return return_dict


def preprocess_inputs(inputs, grid, time):
    """ Process returned xarray dict from supply_inputs to extract required
    cells and year(s). The returned object can be supplied as initial coupling
    input.
    :param inputs: inputs dictionary with input keys (as keys) and
        xarray.DataArray(s) as value
    :type inputs: dict (xarray.DataArray)
    :param grid: numpy.array of shape (ncell, 2) with size 2 being longitude
        and latitude.
    :type: numpy.array
    :param time: years to be extracted
    :type time: int, list(int)
    :return: inputs dictionary with input keys (as keys) and numpy.array(s)
        as value, that CAN DIRECTLY BE SEND TO LPJML AS INPUT (via Coupler)
    :rtype: dict (values as numpy.array)
    """

    # define longitide and latitude DataArray (workaround to reduce dims to
    #   cells)
    lons = xr.DataArray(grid[:, 0], dims="cells")
    lats = xr.DataArray(grid[:, 1], dims="cells")

    # create same format as before but with selected numpy arrays instead of
    #   xarray.DataArray
    input_data = {key: inputs[key].sel(
        longitude=lons, latitude=lats, time=time, method="nearest"
    ).transpose("cells", ...).to_numpy() for key in inputs}

    return input_data
