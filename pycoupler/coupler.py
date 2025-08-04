import os
import sys
import socket
import struct
import tempfile
import copy
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from subprocess import run
from enum import Enum

from pycoupler.config import read_config
from pycoupler.data import (
    LPJmLInputType,
    LPJmLData,
    LPJmLDataSet,
    append_to_dict,
    read_meta,
    read_data,
    read_header,
)
from pycoupler.utils import get_countries


# class for testing purposes
class test_channel:
    """Test channel class for testing purposes"""

    def __init__(self):
        pass

    def sendall(self, val):
        pass

    def recv(self, val):
        pass

    def close(self):
        pass

    def send(self, val):
        pass

    def getsockname(self):
        return ["test", "<none>"]


def write_bytestring_to_file(bytestring, filepath):
    """Write bytestring to file in binary mode (for testing purposes)"""
    with open(filepath, "a") as file:
        file.write(str(bytestring) + "\n")


def read_lines_from_file(filepath):
    """Read lines from a file and return as a list (for testing purposes)"""

    with open(filepath, "r") as file:  # Read the file in binary mode
        bytestring = file.read().strip()

    lines = bytestring.split("\n")
    if int(os.environ["TEST_LINE_COUNTER"]) < len(lines):
        line = lines[int(os.environ["TEST_LINE_COUNTER"])]
        os.environ["TEST_LINE_COUNTER"] = str(
            int(os.environ["TEST_LINE_COUNTER"]) + 1
        )  # noqa
        return line
    else:
        return None


def recvall(channel, size):
    """basic receive function"""
    bytestring = b""
    nbytes = 0
    while nbytes < size:
        bytestring += channel.recv(size - nbytes)
        nbytes += len(bytestring)

    return bytestring


def send_int(channel, val):
    """send integer value to channel"""
    channel.sendall(struct.pack("i", val))


def read_int(channel):
    """read received string as integer"""
    if hasattr(sys, "_called_from_test"):
        # write_bytestring_to_file(inttup[0], os.path.join(os.path.dirname(__file__), '../tests/data/test_receive.txt')) # noqa
        inttup = [
            int(
                read_lines_from_file(
                    f"{os.environ['TEST_PATH']}/data/test_receive.txt"
                )  # noqa
            )
        ]  # noqa
    else:
        intstr = recvall(channel, struct.calcsize("i"))
        inttup = struct.unpack("i", intstr)
    return inttup[0]


def read_short(channel):
    """read received string as short"""
    if hasattr(sys, "_called_from_test"):
        # write_bytestring_to_file(inttup[0], os.path.join(os.path.dirname(__file__), '../tests/data/test_receive.txt')) # noqa
        inttup = [
            int(
                read_lines_from_file(
                    f"{os.environ['TEST_PATH']}/data/test_receive.txt"
                )  # noqa
            )
        ]  # noqa
    else:
        intstr = recvall(channel, struct.calcsize("h"))
        inttup = struct.unpack("h", intstr)
    return inttup[0]


def send_float(channel, val):
    """send float value to channel"""
    channel.sendall(struct.pack("f", val))


def read_float(channel):
    """read received string as float"""
    if hasattr(sys, "_called_from_test"):
        # write_bytestring_to_file(floattup[0], os.path.join(os.path.dirname(__file__), '../tests/data/test_receive.txt')) # noqa
        floattup = [
            float(
                read_lines_from_file(f"{os.environ['TEST_PATH']}/data/test_receive.txt")
            )
        ]  # noqa
    else:
        floatstr = recvall(channel, struct.calcsize("f"))
        floattup = struct.unpack("f", floatstr)

    return floattup[0]


class LPJmlValueType(Enum):
    """Available datatypes"""

    LPJML_BYTE: int = 0
    LPJML_SHORT: int = 1
    LPJML_INT: int = 2
    LPJML_FLOAT: int = 3
    LPJML_DOUBLE: int = 4

    @property
    def type(self):
        """Convert LPJmL data type to Python data types"""
        if self.value > 2:
            return float
        else:
            return int

    @property
    def read_fun(self):
        """Return type correct reading function"""
        if self.name == "LPJML_SHORT":
            read_fun = read_short
        elif self.name == "LPJML_INT":
            read_fun = read_int
        elif self.name == "LPJML_FLOAT":
            read_fun = read_float
        elif self.name == "LPJML_DOUBLE":
            read_fun = read_float
        else:
            raise ValueError(f"lpjml_type {self.name} does not have a read function.")

        return read_fun


class LPJmLToken(Enum):
    """Available tokens"""

    SEND_INPUT: int = 0  # Receiving data from COPAN
    READ_OUTPUT: int = 1  # Sending data to COPAN
    SEND_INPUT_SIZE: int = 2  # Receiving data size from COPAN
    READ_OUTPUT_SIZE: int = 3  # Sending data size to COPAN
    END_COUPLING: int = 4  # Ending communication
    GET_STATUS: int = 5  # Check status of COPAN


def read_token(channel):
    """read integer as LPJmLToken (Enum class)"""
    # create LPJmLToken object
    return LPJmLToken(read_int(channel))


class CopanStatus(Enum):
    """Status of coupled model"""

    COPAN_OK: int = 0
    COPAN_ERR: int = -1


def opentdt(host, port):
    """open channel and validate connection to LPJmL"""
    if hasattr(sys, "_called_from_test"):
        channel = test_channel()
    else:
        # create an INET, STREAMing socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # bind the socket to a public host, and a well-known port
        serversocket.bind((host, port))
        # become a server socket
        serversocket.listen(5)
        # accept connections from outside
        channel, address = serversocket.accept()

    channel.send("1".encode())
    known_int = read_int(channel)  # noqa
    num = read_int(channel)
    num = 1
    send_int(channel, num)
    return channel


class LPJmLCoupler:
    """
    Interface between LPJmL and coupled models via socket communication.

    The LPJmLCoupler initiates a socket channel and follows a specific
    initialization protocol for data exchange with LPJmL.

    Communication Protocol
    ----------------------
    The protocol follows this sequence (→: receive from LPJmL, ←: send to LPJmL):

    1. Initial handshake:
       → version (int32)
       → number_of_cells (int32)
       → number_of_input_streams (int32)
       → number_of_output_streams (int32)

    2. Input stream configuration (for each input stream):
       → SEND_INPUT_SIZE token (int32)
       → stream_index (int32)
       → data_type (int32)
       ← number_of_bands (int32)

    3. Output stream configuration (for each output stream):
       → READ_OUTPUT_SIZE token (int32)
       → stream_index (int32)
       → number_of_steps (int32)
       → number_of_bands (int32)
       → data_type (int32)

    4. Static data exchange (for static outputs like grid):
       → READ_OUTPUT token (int32)
       → output_index (int32)
       → data_array (float/int, shape depends on bands and cells)

    Notes
    -----
    - Protocol version may vary between LPJmL versions
    - Keep the same LPJmLCoupler instance throughout simulation
    - Do not close or reinitialize during simulation

    Parameters
    ----------
    config_file : str
        Path to LPJmL configuration file containing simulation details.
    version : int, default 3
        Coupler protocol version, must match LPJmL internal coupler version.
    host : str, default "localhost"
        Host address where LPJmL server is running.
    port : int, default 2224
        Port number for socket connection.
    """

    def __init__(self, config_file, version=3, host="localhost", port=2224):
        """Constructor method"""

        # read configuration file
        self._config = read_config(config_file)

        if hasattr(self._config, "coupled_host") and hasattr(
            self._config, "coupled_port"
        ):
            if host != "localhost" or port != 2224:
                warnings.warn(
                    "Host and port are set in configuration file. "
                    "Provided host and port are ignored."
                )
            host = self._config.coupled_host
            port = self._config.coupled_port

        # initiate socket connection to LPJmL
        self._init_channel(version, host, port)

        # get input ids based on configuration
        LPJmLInputType.load_config(self._config)

        # initiate coupling, get number of cells, inputs and outputs and verify
        self._init_coupling()

        # communicate status, if valid start LPJmL simulation
        self._communicate_status()

        # create templates for static output data
        self._init_static_data()

        # Read static outputs
        self._iterate_operation(
            length=len(self._static_ids),
            fun=self._read_static_data,
            token=LPJmLToken.READ_OUTPUT,
        )
        # Subtract static inputs from the ones that are read within simulation
        self._noutput_sim -= len(self._static_ids)

        # Create output templates
        self._output_templates = {
            output_key: self._create_xarray_template(int(output_key))
            for output_key in self._output_ids
        }

    # callled when writing class as pickle - exclude channel (socket) attribute
    def __getstate__(self):
        # Create a dictionary of the attributes to pickle, excluding the socket
        state = self.__dict__.copy()
        if "_channel" in state:
            del state["_channel"]  # Exclude the socket
        return state

    @property
    def config(self):
        """Get the underlyig LpjmLConfig object

        Returns
        -------
        LpjmlConfig
            LpjmlConfig object.
        """
        return self._config

    @property
    def ncell(self):
        """Get the number of LPJmL cells for in- and output

        Returns
        -------
        int
            Number of LPJmL cells.
        """
        return self._ncell

    @property
    def operations_left(self):
        """Get the operations left for the current simulation year

        Returns
        -------
        list
            List of operations left for the current simulation year.
        """
        operations = []
        if self._sim_year >= self._config.start_coupling:
            if self._sim_year != self._year_send_input:
                operations.append(LPJmLToken.SEND_INPUT)
        if self._sim_year >= self._config.outputyear:
            if self._sim_year != self._year_read_output:
                operations.append(LPJmLToken.READ_OUTPUT)
        return operations

    @property
    def sim_year(self):
        """Get the current simulation year

        Returns
        -------
        int
            Current simulation year.
        """
        return self._sim_year

    @property
    def sim_years(self):
        """Get a list of all simulation years

        Returns
        -------
        list
            List of all simulation years.
        """
        return [year for year in self.get_sim_years()]

    @property
    def historic_years(self):
        """Get a list of all historic years

        Returns
        -------
        list
            List of all historic years.
        """
        return [year for year in self.get_historic_years(match_period=False)]

    @property
    def coupled_years(self):
        """Get a list of all coupled years

        Returns
        -------
        list
            List of all coupled years.
        """
        return [year for year in self.get_coupled_years(match_period=False)]

    def get_sim_years(self):
        """Get a generator for all simulation years

        Returns
        -------
        generator
            Generator for all simulation years.
        """
        start_year = self._sim_year
        end_year = self._config.lastyear
        current_year = start_year
        while current_year <= end_year:
            yield current_year
            current_year += 1

    def get_historic_years(self, match_period=True):
        """Get a generator for all historic years

        Returns
        -------
        generator
            Generator for all historic years.
        """
        start_year = self._sim_year
        end_year = self.config.start_coupling
        if match_period and start_year >= end_year:
            raise ValueError(
                f"No historic years available. Simulated year {start_year} "
                f"is greater than coupled year {end_year}."
            )
        current_year = start_year
        while current_year < end_year:
            yield current_year
            current_year += 1

    def get_coupled_years(self, match_period=True):
        """Get a generator for all coupled years

        Returns
        -------
        generator
            Generator for all coupled years.
        """
        start_year = self._sim_year
        end_year = self.config.lastyear
        if match_period and (start_year < self.config.start_coupling):
            raise ValueError(
                f"Historic years left. Simulated year {start_year} "
                f"is smaller than coupled_year {end_year}."
            )
        current_year = start_year
        while current_year <= end_year:
            yield current_year
            current_year += 1

    def get_cells(self, id=True):
        """Get a generator for all cells

        Parameters
        ----------
        id : bool, default True
            If True, return cell ids, else return cell indices

        Returns
        -------
        generator
        :rtype: generator
        """
        start_cell = self._config.startgrid
        end_cell = self._config.endgrid
        if id:
            current_cell = start_cell
        else:
            current_cell = 0
            end_cell -= start_cell
        while current_cell <= end_cell:
            yield current_cell
            current_cell += 1

    def code_to_name(self, to_iso_alpha_3=False):
        """Convert country indices to cell names

        Parameters
        ----------
        to_iso_alpha_3 : bool, default False
            If True, convert country indices to ISO alpha-3 codes, else return
            country names
        """
        for static_output in self._static_ids.values():

            if static_output not in ["country", "region"]:
                continue

            getattr(self, static_output).values = getattr(
                self, static_output
            ).values.astype(str)
            name_dict = {
                str(reg["id"]): reg["name"]
                for reg in self._config.to_dict()[f"{static_output}par"]
            }
            if static_output == "country" and to_iso_alpha_3:
                country_dict = get_countries()
                name_dict = {
                    idx: country_dict[reg]["code"] for idx, reg in name_dict.items()
                }
                getattr(self, f"{static_output}").attrs[
                    "long_name"
                ] = f"{static_output} iso alpha-3 code"
            else:
                getattr(self, f"{static_output}").attrs[
                    "long_name"
                ] = f"{static_output} name"

            def replace_values(x):
                return name_dict[x] if x in name_dict else x

            getattr(self, f"{static_output}").values = np.vectorize(
                replace_values
            )(  # noqa
                getattr(self, f"{static_output}").values
            )

    def read_historic_output(self, to_xarray=True):
        """Read historic output from LPJmL

        Parameters
        ----------
        to_xarray : bool, default True
            If True, return output as xarray.DataArray, else return dictionary

        Returns
        -------
        dict or xarray.DataArray
            Dictionary with output keys and corresponding output as numpy arrays
        """
        # read all historic outputs
        hist_years = list()
        for year in self.get_historic_years():
            hist_years.append(year)
            if year == self._config.outputyear:
                output_dict = self.read_output(year=year, to_xarray=False)
            elif year > self._config.outputyear:
                output_dict = append_to_dict(
                    output_dict, self.read_output(year=year, to_xarray=False)
                )

        for key in output_dict:
            if key in output_dict:
                index = [
                    item[0] for item in self._output_ids.items() if item[1] == key
                ][0]
                lpjml_output = self._create_xarray_template(
                    index, time_length=len(hist_years)
                )

                lpjml_output.coords["time"] = pd.date_range(
                    start=str(hist_years[0]), end=str(hist_years[-1] + 1), freq="YE"
                )
                lpjml_output.data = output_dict[key]
                output_dict[key] = lpjml_output

        if to_xarray:
            return LPJmLDataSet(output_dict)
        else:
            return output_dict

    def close(self):
        """Close socket channel"""
        self._channel.close()

    def send_input(self, input_dict, year):
        """Send input data of iterated year as dictionary to LPJmL.

        Parameters
        ----------
        input_dict : dict
            Dictionary of input keys and corresponding input in the form of numpy
            arrays with dimensions (ncell, nband)
        year : int
            Year for which input data is to be sent.

        Returns
        -------
        dict
            Dictionary with input keys and corresponding input in the form of
            numpy arrays with dimensions (ncell, nband)

        Example
        -------
        >>> my_dict = {
        ...     'landuse': <numpy_array_with_shape_(ncell, nband)>,
        ...     'with_tillage': <numpy_array_with_shape_(ncell, nband)>
        ... }
        >>> my_dict
        {'landuse': <numpy_array_with_shape_(ncell, nband)>, 'with_tillage':
        <numpy_array_with_shape_(ncell, nband)>}

        """
        # iteration step check - if number of iterations left exceed simulation
        #   steps (analogous to years left)
        # Year check - if procvided year matches internal simulation year
        if year != self._sim_year:
            raise ValueError(f"Year {year} not matches simulated year {self._sim_year}")
        operations = self.operations_left

        # Check if read_output operation valid
        if not operations or LPJmLToken.SEND_INPUT not in operations:
            raise IndexError(f"No send_input operation left for year {year}")
        elif operations.pop(0) != LPJmLToken.SEND_INPUT:
            raise IndexError(f"Invalid operation order. Expected read_output")

        # iterate over outputs for private send_input_data
        self._iterate_operation(
            length=self._ninput,
            fun=self._send_input_data,
            token=LPJmLToken.SEND_INPUT,
            args={"data": input_dict, "year": year},
        )

        self._year_send_input = year

        # check if all operations have been performed and increase sim_year
        if not self.operations_left:
            self._sim_year += 1

    def read_output(self, year, to_xarray=True):
        """Read LPJmL output data of the specified year.

        Parameters
        ----------
        year : int
            Year for which output data is to be read.
        to_xarray : bool, default True
            If True, output is returned as xarray.DataArray.

        Returns
        -------
        :return: Dictionary with output id/name as keys and outputs in the form
                of numpy.array or xarray.DataArray.
        :rtype: dict or xarray.DataArray
        """
        # Validate the year
        if year != self._sim_year:
            raise ValueError(
                f"Year {year} does not match simulated year {self._sim_year}"
            )

        # Check if read_output operation is valid
        if (
            not self.operations_left
            or self.operations_left[0] != LPJmLToken.READ_OUTPUT
        ):
            raise IndexError(f"No read_output operation left for year {year}")

        all_output = copy.deepcopy(self._output_templates)
        if to_xarray:
            output = {
                key: all_output[key]
                for key in all_output.keys()
                if key not in self._static_ids
            }  # noqa
            time = pd.date_range(str(year), periods=1, freq="YE")
            for value in output.values():
                value.coords["time"] = time
        else:
            output = {
                key: all_output[key].values
                for key in all_output.keys()
                if key not in self._static_ids
            }  # noqa

        output_iterations = sum(steps for steps in self._output_steps if steps > 0)

        # Perform subannual (monthly, daily) read_output operation
        output = self._iterate_operation(
            length=output_iterations,
            fun=self._read_output_data,
            token=LPJmLToken.READ_OUTPUT,
            args={"output": output, "year": year, "to_xarray": to_xarray},
        )

        # Update the dictionary keys
        for old_key, new_key in self._output_ids.items():
            if old_key in output:
                output[new_key] = output.pop(old_key)

        if to_xarray:
            output = LPJmLDataSet(output)

        self._year_read_output = year
        self._output_count_steps = [0] * len(self._output_count_steps)
        # If all operations have been performed, increase sim_year
        if not self.operations_left:
            self._sim_year += 1

        return output

    def read_input(self, start_year=None, end_year=None, copy=True):
        """Read coupled input data from netcdf files and copy them to the
        simulation directory if copy=True.

        If no start_year and end_year are provided, the default start_year and
        end_year from the config are used.

        Parameters
        ----------
        start_year : int, default None
            Start year of input data.
        end_year : int, default None
            End year of input data.
        copy : bool, default True
            If True, input data is copied to simulation directory

        Returns
        -------
        LPJmLDataSet
            LPJmLDataSet with input data with input names as keys
        """
        if copy:
            self._copy_input(start_year=start_year, end_year=end_year)
        # read coupled input data from netcdf files (as xarray.DataArray)
        inputs = {}
        for key in self.config.get_input_sockets(id_only=True):
            inputs[key] = self.assign_config_band_names(
                read_data(
                    f"{self._config.sim_path}/input/{key}.nc",
                    var_name=key,
                    multiple_bands=True,
                ),
                key=key,
            )
            # TODO: add support for input meta data file (if available)
            # inputs[key].add_meta() to replace workaround with assign_band_names

        # if no start_year and end_year provided and only one year is supplied
        #   ensure years are the same (although they are not - but to avoid
        #   errors in LPJmLDataSet handling)
        if not start_year and not end_year and len(inputs) > 1:
            year = max(
                inputs.values(), key=lambda inp: inp.time.item()
            ).time.item()  # noqa
            for key in inputs.keys():
                inputs[key] = inputs[key].drop("time").assign_coords({"time": [year]})

        inputs = LPJmLDataSet(inputs)
        # define longitide and latitude DataArray (workaround to reduce dims to
        #   cells)
        if not hasattr(self, "_cached_grid"):
            self._cached_grid = {
                "lons": xr.DataArray(self.grid[:, 0], dims="cell"),
                "lats": xr.DataArray(self.grid[:, 1], dims="cell"),
            }

        lons = self._cached_grid["lons"]
        lats = self._cached_grid["lats"]

        if start_year and end_year:
            kwargs = {"time": [year for year in range(start_year, end_year + 1)]}
        elif start_year and not end_year:
            kwargs = {
                "time": [
                    year for year in range(start_year, max(inputs.time.values) + 1)
                ]
            }
        elif not start_year and end_year:
            kwargs = {
                "time": [year for year in range(min(inputs.time.values), end_year + 1)]
            }
        else:
            kwargs = {}

        inputs.coords["time"] = pd.date_range(
            start=str(min(inputs.coords["time"].values)),
            end=str(max(inputs.coords["time"].values) + 1),
            freq="YE",
        )
        # create same format as before but with selected numpy arrays instead
        # of xarray.DataArray
        inputs = (
            inputs.sel(lon=lons, lat=lats, method="nearest", **kwargs)
            .transpose("cell", ..., "time")
            .load()
        )

        return inputs

    def assign_config_band_names(self, x, key):
        irr_systems_short = ["rainfed", "irrigated"]
        irr_systems_long = [
            "rainfed",
            "surface irrigated",
            "sprinkler irrigated",
            "drip irrigated",
        ]

        if "band" in x.dims:
            band_dim = "band"
        else:
            band_dim = f"band ({key})"

        band_names = x[band_dim]
        if len(band_names) in [1, 12, 365]:
            y = x.assign_coords(
                {band_dim: [str(i + 1) for i in range(len(band_names))]}
            )
        elif (
            len(band_names) % len(self.config.cftmap) == 0
            or len(band_names) % len(self.config.landusemap) == 0
        ):
            if len(band_names) % len(self.config.cftmap) == 0:
                len_irr_systems = len(band_names) // len(self.config.cftmap)
            else:
                len_irr_systems = len(band_names) // len(self.config.landusemap)

            if len_irr_systems == 2:
                irr_systems = irr_systems_short
            elif len_irr_systems == 4:
                irr_systems = irr_systems_long
            else:
                return x

            if len(band_names) % len(self.config.cftmap) == 0:
                combined_list = [
                    f"{irr} {crop}"
                    for irr in irr_systems
                    for crop in self.config.cftmap
                ]
            else:
                combined_list = [
                    f"{irr} {crop}"
                    for irr in irr_systems
                    for crop in self.config.landusemap
                ]  # {irr}

            y = x.assign_coords({band_dim: combined_list})
        return y

    def _copy_input(self, start_year, end_year):
        """Copy and convert and save input files as NetCDF4 files to input
        directory for selected years to make them easily readable as well as to
        avoid large file sizes.
        """
        input_path = f"{self._config.sim_path}/input"
        # get defined input sockets
        if not os.path.isdir(input_path):
            os.makedirs(input_path)
            print(f"Created input path '{input_path}'")

        sock_inputs = self.config.get_input_sockets()

        # utility function to get general temp folder for every system
        temp_dir = tempfile.gettempdir()

        if start_year is None:
            start_year = self.config.start_coupling - 1
        if end_year is None:
            end_year = self.config.start_coupling - 1
        if start_year > end_year:
            raise ValueError("start_year cannot be later than end_year")

        # iterate over each inputs to be send via sockets (get initial values)
        for key in sock_inputs:
            # check if working on the cluster (workaround by Ciaron)
            #   (might be adjusted to the new cluster coming soon ...)
            if self.config.inpath and (not sock_inputs[key]["name"].startswith("/")):
                sock_inputs[key][
                    "name"
                ] = f"{self.config.inpath}/{sock_inputs[key]['name']}"
            # get input file name
            file_name_clm = sock_inputs[key]["name"].split("/")[-1]
            # name tmp file after original name (even though could be random)
            file_name_tmp = f"{file_name_clm.split('.')[0]}_tmp.clm"

            if not hasattr(sys, "_called_from_test"):
                # read meta data of input file
                meta_data = read_header(sock_inputs[key]["name"])
            else:
                meta_data = read_meta(
                    f"{os.environ['TEST_PATH']}/data/input/{key}.nc.json"
                )

            # determine start cut off and end cut off year
            if meta_data.firstyear > end_year:
                # data is after simulation period
                cut_start_year = meta_data.firstyear
                cut_end_year = meta_data.firstyear
            elif meta_data.lastyear < start_year:
                # data is before simulation period
                cut_start_year = meta_data.lastyear
                cut_end_year = meta_data.lastyear
            elif meta_data.firstyear >= start_year:
                # data begins in simulation period
                cut_start_year = meta_data.firstyear
                cut_end_year = min(meta_data.lastyear, end_year)
            elif meta_data.lastyear < end_year:
                # data ends in simulation period
                cut_start_year = start_year
                cut_end_year = meta_data.lastyear
            elif meta_data.firstyear <= start_year and meta_data.lastyear >= start_year:
                # data starts before simulation period,
                # but simulation is within data period
                cut_start_year = start_year
                cut_end = cut_end_year = min(meta_data.lastyear, end_year)

            cut_clm_start = [
                f"{self._config.model_path}/bin/cutclm",
                str(cut_start_year),
                sock_inputs[key]["name"],
                f"{temp_dir}/1_{file_name_tmp}",
            ]
            if not hasattr(sys, "_called_from_test"):
                run(cut_clm_start, stdout=open(os.devnull, "wb"))

            # predefine cut clm command for reusage
            # cannot deal with overwriting a temp file with same name
            cut_clm_end = [
                f"{self._config.model_path}/bin/cutclm",
                "-end",
                str(cut_end_year),
                f"{temp_dir}/1_{file_name_tmp}",
                f"{temp_dir}/2_{file_name_tmp}",
            ]
            if not hasattr(sys, "_called_from_test"):
                run(cut_clm_end, stdout=open(os.devnull, "wb"))

            # a flag for multi (categorical) band input - if true, set
            #   "-landuse"
            if LPJmLInputType(name=key).has_bands:
                is_multiband = "-landuse"
            else:
                is_multiband = None

            # a flag for integer input - if true, set "-int"
            if LPJmLInputType(name=key).type == int:
                is_int = "-intnetcdf"
            else:
                is_int = None

            # default grid file (only valid for 0.5 degree inputs)
            if self.config.input.coord.name.startswith("/"):
                grid_file = self.config.input.coord.name
            else:
                grid_file = f"{self.config.inpath}/{self.config.input.coord.name}"
            # convert clm input to netcdf files
            conversion_cmd = [
                f"{self._config.model_path}/bin/clm2cdf",
                is_int,
                is_multiband,
                key,
                grid_file,
                f"{temp_dir}/2_{file_name_tmp}",
                f"{input_path}/{key}.nc",
            ]

            if None in conversion_cmd:
                conversion_cmd.remove(None)

            if not hasattr(sys, "_called_from_test"):
                run(conversion_cmd)
            else:
                return "tested"
            # remove the temporary clm (binary) files, 1_* is not created in
            #   every case
            if os.path.isfile(f"{temp_dir}/1_{file_name_tmp}"):
                os.remove(f"{temp_dir}/1_{file_name_tmp}")
            if os.path.isfile(f"{temp_dir}/2_{file_name_tmp}"):
                os.remove(f"{temp_dir}/2_{file_name_tmp}")

    def _init_channel(self, version, host, port):
        # open/initialize socket channel
        self._channel = opentdt(host, port)
        # Check coupler protocol version
        self.version = read_int(self._channel)
        if self.version != version:
            self.close()
            raise ValueError(
                f"Invalid coupler version {version}, must be {self.version}"
            )

    def _init_coupling(self):
        """Initialize coupling"""
        # initiate simulation time
        self._sim_year = min(self._config.outputyear, self._config.start_coupling)
        self._year_read_output = None
        self._year_send_input = None

        # read amount of LPJml cells
        self._ncell = read_int(self._channel)
        if self._ncell != len(range(self._config.startgrid, self._config.endgrid + 1)):
            self.close()
            raise ValueError(
                f"Invalid number of cells received ({self._ncell}), must be"
                f" {self._config.ncell} according to configuration."
            )

        # read amount of input streams
        self._ninput = read_int(self._channel)

        # read amount of output streams
        self._noutput = self._noutput_sim = read_int(self._channel)

        # verify and initialize input attributes and send input information
        self._init_input()

        # verify and initialize output attributes and read output information
        self._init_output()

    def _init_input(self):
        """Initialize input"""
        input_sockets = self._config.get_input_sockets()

        if self._ninput != len(input_sockets):
            self.close()
            raise ValueError(
                f"Invalid number of input streams received ({self._ninput}),"
                f" must be {len(input_sockets)} according to"
                f" configuration."
            )
        else:
            # init lists to be filled with nbands, types per output
            # account for not represented input files (+ 42)
            self._input_types = [-1] * (len(self.config.input.__dict__) + 42)

            # get input indices
            self._input_ids = {
                input_sockets[inp]["id"]: inp
                for inp in input_sockets
                if input_sockets[inp]["id"] in LPJmLInputType.ids
            }

            # send number of bands for each output data stream
            self._iterate_operation(
                length=self._ninput,
                fun=self._send_band_size,
                token=LPJmLToken.SEND_INPUT_SIZE,
                args={"input_bands": self._get_config_input_sockets()},
            )

    def _init_output(self):
        """Initialize output"""
        output_sockets = self._config.get_output_sockets()

        if self._noutput != len(output_sockets):
            self.close()
            raise ValueError(
                f"Invalid number of output streams received ({self._noutput})"
                f", must be {len(output_sockets)} according to"
                f" configuration."
            )

        else:
            # init lists to be filled with nbands, nsteps per output
            len_outputvar = len(self.config.outputvar)
            self._output_bands = [-1] * len_outputvar
            self._output_steps = [-1] * len_outputvar
            self._output_count_steps = [0] * len_outputvar
            self._output_types = [-1] * len_outputvar

            # Get output indices
            self._output_ids = {}
            self._static_ids = {}
            for out in output_sockets:
                self._output_ids[output_sockets[out]["index"]] = out
                if output_sockets[out]["id"] in [
                    "grid",
                    "country",
                    "region",
                    "terr_area",
                    "lake_area",
                ]:
                    self._static_ids[output_sockets[out]["index"]] = out

            # Get number of bands per cell for each output data stream
            self._iterate_operation(
                length=self._noutput,
                fun=self._read_output_details,
                token=LPJmLToken.READ_OUTPUT_SIZE,
            )

    def _iterate_operation(self, length, fun, token, args=None):
        """Iterate reading/sending operation for sequence of inputs and/or
        outputs
        """

        for _ in range(length):
            # check token
            self._check_token(token=token)

            # execute method on channel and if supplied further method arguments
            output = fun(**args) if args else fun()
            if args is not None and "output" in args.keys():
                args["output"] = output

        if args is not None and "output" in args.keys():
            return output
        else:
            None

    def _check_token(self, token):
        """check if read token matches the expected token"""

        # check if read token matches expected token and return read token
        received_token = read_token(self._channel)

        if received_token is not token:
            self.close()
            raise ValueError(
                f"Received LPJmLToken {received_token.name} is not {token.name}"
            )  # noqa

    def _check_index(self, index):
        """check if read index matches the expected token"""

        received_index = read_int(self._channel)
        if received_index != index:
            self.close()
            raise ValueError(
                f"The expected index: {index} does not "
                + f"match the received index: {received_index}"
            )

    def _check_year(self, year):
        """check if read year matches the expected token"""

        received_year = read_int(self._channel)
        if received_year != year:
            self.close()
            raise ValueError(
                f"The expected year: {year} does not "
                + f"match the received year: {received_year}"
            )

    def _send_band_size(self, input_bands):
        """Send input band size for read index to socket"""
        index = read_int(self._channel)
        # convert received LPJmL data types into Python compatible types
        self._set_input_types(index)
        if index in input_bands.keys():
            # Send number of bands
            send_int(self._channel, val=input_bands[index])
        else:
            self.close()
            raise ValueError(f"Input of input ID {index} not supported.")

    def _set_input_types(self, index):
        """Convert received LPJmL data types into Python compatible types"""
        self._input_types[index] = LPJmlValueType(read_int(self._channel))

    def _get_config_input_sockets(self):
        """Get and validate input sockets, check if defined in LPJmLInputType
        class. If not add to LPJmLInputType.
        """
        # get defined input sockets
        sockets = self._config.get_input_sockets()
        socket_ids = [sock_id["id"] for sock_id in sockets.values()]
        # filter input names
        input_ids = [inp for inp in LPJmLInputType.ids]

        # check if input is defined in LPJmLInputType (band size required)
        valid_inputs = {
            LPJmLInputType(id=sock_id).id: LPJmLInputType(id=sock_id).nband  # noqa
            for sock_id in socket_ids
            if sock_id in input_ids
        }
        if len(socket_ids) != len(valid_inputs):
            self.close()
            raise ValueError(
                f"Configurated sockets {sockets.keys()} not defined in"
                + f" {LPJmLInputType.names}!"
            )
        return valid_inputs

    def _read_output_details(self):
        """Read output details per output index (timesteps, number of bands,
        data types) from socket
        """
        index = read_int(self._channel)
        # Get number of steps for output
        self._output_steps[index] = read_int(self._channel)
        # Get number of bands for output
        self._output_bands[index] = read_int(self._channel)
        # Get datatype for output
        self._output_types[index] = LPJmlValueType(read_int(self._channel))

        send_int(self._channel, CopanStatus.COPAN_OK.value)

    def _communicate_status(self):
        """Check if LPJmL token equals GET_STATUS, send OK or ERROR"""
        check_token = LPJmLToken(read_int(self._channel))
        if check_token == LPJmLToken.GET_STATUS:
            if self._ninput != 0 and self._noutput != 0:
                send_int(self._channel, CopanStatus.COPAN_OK.value)
            else:
                self.close()
                send_int(self._channel, CopanStatus.COPAN_ERR.value)
                raise ValueError("No inputs OR outputs defined.")
        else:
            raise ValueError(
                f"Got LPJmLToken {check_token.name}, though "
                + f"LPJmLToken {LPJmLToken.GET_STATUS} expected."
            )

    def _init_static_data(self):

        for static_id in self._static_ids:

            # Create empty array for of corresponding type
            tmp_static = np.zeros(
                shape=(self._ncell, self._output_bands[static_id]),
                dtype=self._output_types[static_id].type,
            )

            # Fill array with missing values
            if self._output_types[static_id].type == int:
                tmp_static[:] = -9999
            else:
                tmp_static[:] = np.nan

            # grid data is handled differently with coords being assigned
            if self._static_ids[static_id] == "grid":
                tmp_static = LPJmLData(
                    data=tmp_static,
                    dims=("cell", "band"),
                    coords=dict(
                        cell=np.arange(
                            self._config.startgrid, self._config.endgrid + 1
                        ),
                        band=["lon", "lat"],
                    ),
                    name="grid",
                )

            # other static data is handled like common output data without time
            else:
                tmp_static = LPJmLData(
                    data=tmp_static,
                    dims=("cell", "band"),
                    coords=dict(
                        cell=np.arange(
                            self._config.startgrid, self._config.endgrid + 1
                        ),
                        band=np.arange(self._output_bands[static_id]),
                    ),
                    name=self._static_ids[static_id],
                )

            setattr(self, f"{self._static_ids[static_id]}", tmp_static)

    def _read_static_data(self):
        """Read static data to be called within initialization of coupler.
        Currently only grid data supported
        """
        index = read_int(self._channel)

        read_fun = self._output_types[index].read_fun
        meta_data = self._read_meta_output(index)

        # iterate over ncell and read + assign lon and lat values
        for cell in range(0, self._ncell):
            for band in range(0, self._output_bands[index]):
                getattr(self, f"{self._static_ids[index]}")[cell, band] = (
                    read_fun(self._channel) * meta_data.scalar
                )

        # add meta data to xarray
        getattr(self, f"{self._static_ids[index]}").add_meta(meta_data)

        # add longitude and latitude coodinates to xarray
        getattr(self, f"{self._static_ids[index]}").coords["lon"] = (
            ("cell",),
            self.grid.data[:, 0],
        )
        getattr(self, f"{self._static_ids[index]}").coords["lat"] = (
            ("cell",),
            self.grid.data[:, 1],
        )

    def _create_xarray_template(self, index, time_length=1):
        """Create xarray template for output data"""
        bands = self._output_bands[index]
        steps_as_bands = False
        if bands == 1:
            if self._output_steps[index] > 1:
                bands = self._output_steps[index]
                steps_as_bands = True

        # create output numpy array template to be filled with output
        output_tmpl = np.zeros(
            shape=(self._ncell, bands, time_length),  # time = 1
            dtype=self._output_types[index].type,
        )

        # Check if data array is of type integer, use -9999 for nan
        if self._output_types[index].type == int:
            output_tmpl[:] = -9999
        else:
            output_tmpl[:] = np.nan

        # create xarray data array with correct dimensions and coord
        output_tmpl = LPJmLData(
            data=output_tmpl,
            dims=("cell", "band", "time"),
            coords=dict(
                cell=np.arange(self._config.startgrid, self._config.endgrid + 1),
                lon=(["cell"], self.grid.coords["lon"].values),
                lat=(["cell"], self.grid.coords["lat"].values),
                band=np.arange(bands),  # [str(i) for i in range(bands)],
                time=np.arange(time_length),
            ),
            name=self._output_ids[index],
        )
        # read output values
        if self._config.output_metafile:
            meta_output = self._read_meta_output(index=index)
        else:
            meta_output = None

        if meta_output:
            # add meta information to output
            output_tmpl.add_meta(meta_output)

            output_tmpl = output_tmpl.rename(band=f"band ({self._output_ids[index]})")
            # add meta data to output
        return output_tmpl

    def _send_input_data(self, data, year):
        """Send input data checks supplied object type, object dimensions data
        format and year for input index. If set correct executes private
        send_input_values method that does the sending.
        """
        index = read_int(self._channel)
        if isinstance(data, LPJmLDataSet):
            data = data.to_numpy()
        elif not isinstance(data[self._input_ids[index]], np.ndarray):
            self.close()
            raise TypeError(
                "Unsupported object type. Please supply a numpy "
                + "array with the dimension of (ncells, nband)."
            )

        # type check conversion
        if self._input_types[index].type == float:
            type_check = np.floating
        elif self._input_types[index].type == int:
            type_check = np.integer

        if not np.issubdtype(data[self._input_ids[index]].dtype, type_check):
            self.close()
            raise TypeError(
                f"Unsupported type: {data[self._input_ids[index]].dtype} "
                + "Please supply a numpy array with data type: "
                + f"{np.dtype(self._input_types[index].type)}."
            )

        # check received year
        self._check_year(year)

        if index in self._input_ids.keys():
            # get corresponding number of bands from LPJmLInputType class
            bands = LPJmLInputType(id=index).nband
            if not np.shape(data[self._input_ids[index]]) == (self._ncell, bands):
                if (
                    bands == 1
                    and not np.shape(data[self._input_ids[index]])[0] == self._ncell
                ):  # noqa
                    self.close()
                    raise ValueError(
                        "The dimensions of the supplied data: "
                        + f"{(self._ncell, bands)} does not match the "
                        + f"needed dimensions for {self._input_ids[index]}"
                        + f": {(self._ncell, bands)}."
                    )
            # execute sending values method to actually send the input to
            #   socket
            self._send_input_values(data[self._input_ids[index]])

    def _send_input_values(self, data):
        """Iterate over all values to be sent to the socket. Recursive iteration
        with the correct order of bands and cells for inputs.
        """
        dims = data.shape
        one_band = len(dims) == 1
        cells, bands = dims[0], dims[1]

        # Determine the send function based on data type
        send_function = send_int if "int" in str(data.dtype) else send_float

        # Iterate over bands (outer loop) and cells (inner loop)
        for cell in range(cells):
            for band in range(bands):
                # Send the value to the socket
                if one_band:
                    send_function(self._channel, data[cell].item())
                else:
                    send_function(self._channel, data[cell, band].item())

    def _read_output_data(self, output, year, to_xarray=True):
        index = read_int(self._channel)
        self._check_year(year)

        steps_as_bands = False
        if self._output_steps[index] > 1:
            if self._output_bands[index] == 1:
                bands = self._output_steps[index]
                steps_as_bands = True
            else:
                raise ValueError(
                    "Subannual output data with more than one band not supported."
                )
        if not to_xarray:
            # read and assign corresponding values from socket to numpy array
            output[index] = self._read_output_values(
                output=output[index],
                index=index,
                dims=list(np.shape(output[index])),
                steps_as_bands=steps_as_bands,
            )
        else:
            # read and assign corresponding values from socket to numpy array
            output[index].values = self._read_output_values(
                output=output[index].values,
                index=index,
                dims=list(np.shape(output[index])),
                steps_as_bands=steps_as_bands,
            )

        # as list for appending/extending as list
        return output

    def _read_output_values(self, output, index, dims=None, steps_as_bands=False):
        """Iterate over all values to be read from the socket. Recursive iteration
        with the correct order of cells and bands for outputs.
        """
        cells, bands = dims[0], dims[1]
        # Determine if there is only one band
        one_band = cells > 0 and bands == 0

        # Iterate over cells (outer loop) and bands (inner loop)
        for band in range(bands):
            for cell in range(cells):
                # Read the value from the socket
                if one_band:
                    output[cell] = self._output_types[index].read_fun(self._channel)
                elif steps_as_bands:
                    output[cell, self._output_count_steps[index]] = self._output_types[
                        index
                    ].read_fun(self._channel)
                else:
                    output[cell, band] = self._output_types[index].read_fun(
                        self._channel
                    )
            if steps_as_bands:
                self._output_count_steps[index] += 1
                break

        return output

    def _read_meta_output(self, index=None, output_id=None):
        """Read meta output data from socket. Returns dictionary with
        corresponding meta output id and value.
        """
        if output_id is not None:
            output = self._config.output[output_id]
        elif index is not None:
            output = [
                out for out in self._config.output if out.id == self._output_ids[index]
            ][0]
        else:
            raise ValueError("Either index or output_id must be supplied")

        # Read meta data from of corresponding output id using config info
        lpjml_meta = read_meta(f"{output.file.name}.json")

        # Set meta data correct for coupling format
        lpjml_meta.timestep = 1
        lpjml_meta.format = "sock"

        # Remove unnecessary attributes
        delattr(lpjml_meta, "bigendian")
        delattr(lpjml_meta, "filename")

        return lpjml_meta

    def __repr__(self):
        """Representation of the Coupler object"""
        if hasattr(self, "_channel"):
            try:
                port = self._channel.getsockname()[1]
            except OSError:
                port = "<closed>"
        else:
            port = "<none>"

        summary = f"<pycoupler.{self.__class__.__name__}>"
        summary = "\n".join(
            [
                summary,
                f"Simulation:  (version: {self.version}, localhost:{port})",
                f"  * sim_year   {min(self._sim_year, self._config.lastyear)}",
                f"  * ncell      {self._ncell}",
                f"  * ninput     {self._ninput}",
            ]
        )
        summary = "\n".join([summary, self._config.__repr__(sub_repr=1)])

        return summary
