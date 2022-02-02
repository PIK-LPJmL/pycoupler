import os
import tempfile
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
                  start_year=None, end_year=None):
    """Convert and save input files as NetCDF4 files to input directory for
    selected years to avoid large file sizes
    """
    # get defined input sockets
    config = read_config(config_file)
    sock_inputs = config.get_input_sockets().keys()
    # get input paths for historic run
    historic_config = read_config(file_name=historic_config_file)
    sock_inputs = historic_config.get_inputs(id_only=False, inputs=sock_inputs)
    # collect via key value pairs
    return_dict = {}
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
        # get input file name
        file_name_clm = sock_inputs[key]['name'].split("/")[-1]
        # name tmp file after original name (even though could be random)
        file_name_tmp = (
            f"{file_name_clm.split('.')[0]}_tmp.clm"
        )
        # predefine cut clm command for reusage
        cut_clm_start = [f"{model_path}/bin/cutclm",
                         str(start_year), sock_inputs[key]['name'],
                         f"{tempfile.gettempdir()}/2_{file_name_tmp}"]
        if start_year and not end_year:
            # run cut clm file before start year
            run(cut_clm_start)
        elif not start_year and end_year:
            # run cut clm file after end year
            run([f"{model_path}/bin/cutclm",
                 "-end", str(end_year), sock_inputs[key]['name'],
                 f"{tempfile.gettempdir()}/2_{file_name_tmp}"])

        else:
            # run cut clm file before start year and after end year in sequence
            run(cut_clm_start)
            # cannot deal with overwriting a temp file with same name
            cut_clm_end = [f"{model_path}/bin/cutclm",
                           "-end", str(end_year),
                           f"{tempfile.gettempdir()}/1_{file_name_tmp}",
                           f"{tempfile.gettempdir()}/2_{file_name_tmp}"]
            run(cut_clm_end)
        # a flag for multi (categorical) band input - if true, set "-landuse"
        if getattr(Inputs, key).bands:
            xarg = "-landuse"
        else:
            xarg = ""
        # default grid file (only valid for 0.5 degree inputs)
        grid_file = (
            "/p/projects/lpjml/input/historical/input_VERSION2/grid.bin"
        )
        # create file name with extension to inform about the extracted time
        #   span
        file_name_nc = (
            f"{file_name_clm.split('.')[0]}_{start_year}-{end_year}.nc"
        )
        # convert clm input to netcdf files
        run([f"{model_path}/bin/clm2cdf", xarg, key, grid_file,
             f"{tempfile.gettempdir()}/2_{file_name_tmp}",
             f"{input_path}/{file_name_nc}"])
        # remove the temporary clm (binary) files, 1_* is not created in every
        #   case
        if os.path.isfile(f"{tempfile.gettempdir()}/{file_name_tmp}"):
            os.remove(f"{tempfile.gettempdir()}/1_{file_name_tmp}")
        os.remove(f"{tempfile.gettempdir()}/2_{file_name_tmp}")
        # collect created input filenames and connect with input key
        return_dict[key] = file_name_nc
    return return_dict
