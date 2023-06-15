import os
import socket
import struct
import tempfile
import numpy as np
import pandas as pd
import xarray as xr

from subprocess import run, CalledProcessError
from enum import Enum

from pycoupler.config import read_config
from pycoupler.data import LPJmLInputType, LPJmLData, LPJmLDataSet,\
    append_to_dict, read_meta, read_data


def recvall(channel, size):
    """basic receive function
    """
    bytestring = b''
    nbytes = 0
    while nbytes < size:
        bytestring += channel.recv(size-nbytes)
        nbytes += len(bytestring)
    return bytestring


def send_int(channel, val):
    """send integer value to channel
    """
    channel.sendall(struct.pack('i', val))


def read_int(channel):
    """read received string as integer
    """
    intstr = recvall(channel, struct.calcsize('i'))
    inttup = struct.unpack('i', intstr)
    return inttup[0]


def read_short(channel):
    """read received string as short
    """
    intstr = recvall(channel, struct.calcsize('h'))
    inttup = struct.unpack('h', intstr)
    return inttup[0]


def send_float(channel, val):
    """send float value to channel
    """
    channel.sendall(struct.pack('f', val))


def read_float(channel):
    """read received string as float
    """
    floatstr = recvall(channel, struct.calcsize('f'))
    floattup = struct.unpack('f', floatstr)
    return floattup[0]


class LPJmlValueType(Enum):
    """Available datatypes
    """
    LPJML_BYTE: int = 0
    LPJML_SHORT: int = 1
    LPJML_INT: int = 2
    LPJML_FLOAT: int = 3
    LPJML_DOUBLE: int = 4

    @property
    def type(self):
        """Convert LPJmL data type to Python data types
        """
        if self.value > 2:
            return float
        else:
            return int

    @property
    def read_fun(self):
        """Return type correct reading function
        """
        if self.name == "LPJML_SHORT":
            read_fun = read_short
        elif self.name == "LPJML_INT":
            read_fun = read_int
        elif self.name == "LPJML_FLOAT":
            read_fun = read_float
        elif self.name == "LPJML_DOUBLE":
            read_fun = read_float
        else:
            raise ValueError(
                f"lpjml_type {self.name} does not have a read function."
            )

        return read_fun


class LPJmLToken(Enum):
    """Available tokens"""
    SEND_INPUT: int = 0       # Receiving data from COPAN
    READ_OUTPUT: int = 1       # Sending data to COPAN
    SEND_INPUT_SIZE: int = 2  # Receiving data size from COPAN
    READ_OUTPUT_SIZE: int = 3  # Sending data size to COPAN
    END_COUPLING: int = 4       # Ending communication
    GET_STATUS: int = 5     # Check status of COPAN


def read_token(channel):
    """read integer as LPJmLToken (Enum class)
    """
    # create LPJmLToken object
    return LPJmLToken(read_int(channel))


class CopanStatus(Enum):
    """Status of copan:CORE"""
    COPAN_OK: int = 0
    COPAN_ERR: int = -1


def opentdt(host, port):
    """open channel and validate connection to LPJmL
    """
    # create an INET, STREAMing socket
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # bind the socket to a public host, and a well-known port
    serversocket.bind((host, port))
    # become a server socket
    serversocket.listen(5)
    # accept connections from outside
    channel, address = serversocket.accept()
    channel.send('1'.encode())
    known_int = read_int(channel)
    num = read_int(channel)
    num = 1
    send_int(channel, num)
    return channel


class LPJmLCoupler:
    """ LPJmLCoupler class serves as the interface between LPJmL and the model
    to couple with. It initiates a socket channel, provides it and follows the
    initialization protocol (->: receiving, <-: sending, for: iteration):
    -> version (int32)
    -> number of cells (int32)
    -> number of input streams (int32)
    -> number of output streams (int32)
    for <number of input streams>:
        -> SEND_INPUT_SIZE (int32)
        -> index (int32)
        -> type(int32)
        <- number of bands (int32)
    for <number of output streams>:
        -> READ_OUTPUT_SIZE (int32)
        -> index (int32)
        -> number of steps(int32)
        -> number of bands (int32)
        -> type(int32)
    for <number of static outputs (config)>:
        -> READ_OUTPUT (int_32)
        -> index (int32)
        if <static output == grid>:
            -> data(cell1, band1), …, data(celln, band1)
               data(cell1, band2), …, data(celln, band2)
        else:
            <next>
    The protocol may vary from version to version hence this only reflects the
    current state of the protocol. Please use the LPJmLCoupler Instance
    throughout the Simulation and do not close or reinitiate in between.

    :param config_file: file name (including relative/absolute path) of the
        corresponding LPJmL configuration to be read simulation details from
    :type config_file: str
    :param version: version of the coupler, to be validated with LPJmL internal
        coupler
    :type version: int
    :param host: host address of the server LPJmL is running. Defaults to ""
        (all IPs of localhost)
    :type host: str
    :param port: port of the server address. Defaults to 2224
    :type port: int
    """
    def __init__(self, config_file, version=3, host='', port=2224):
        """Constructor method
        """
        self.__config = read_config(config_file)
        self.__sim_year = min(self.__config.outputyear,
                              self.__config.start_coupling)
        self.__year_read_output = None
        self.__year_send_input = None

        # open/initialize socket channel
        self.__channel = opentdt(host, port)
        # Check coupler protocol version
        self.version = read_int(self.__channel)
        if self.version != version:
            self.close()
            raise ValueError(
                f"Invalid coupler version {version}, must be {self.version}"
            )
        # read amount of LPJml cells
        self.__ncell = read_int(self.__channel)

        # read amount of input streams
        self.__ninput = read_int(self.__channel)

        # read amount of output streams
        self.__noutput = self.__noutput_sim = read_int(self.__channel)

        # init list to be filled with types per output
        self.__input_types = [-1] * len(self.config.input.__dict__)
        # Send number of bands per cell for each input data stream
        self.__input_bands = self.__get_config_input_sockets()

        # Send number of bands for each output data stream
        self.__iterate_operation(
            length=self.__ninput, fun=self.__send_band_size,
            token=LPJmLToken.SEND_INPUT_SIZE,
        )

        # init list to be filled with nbands, nsteps per output
        self.__output_bands = [-1] * len(self.config.outputvar)
        self.__output_steps = [-1] * len(self.config.outputvar)
        self.__output_types = [-1] * len(self.config.outputvar)

        # init counter to be filled with number of static outputs
        self.__noutput_static = 0

        # Check for static output
        outputs_avail = self.__config.get_output_avail(id_only=False)

        # Get output ids
        self.__globalflux_id = [out["id"] for out in outputs_avail if out[
            "name"] == "globalflux"][0]
        self.__grid_id = [out["id"] for out in outputs_avail if out[
            "name"] == "grid"][0]
        self.__static_id = [out["id"] for out in outputs_avail if out[
            "name"] in ["grid", "country", "region"]]

        # Get number of bands per cell for each output data stream
        self.__iterate_operation(length=self.__noutput,
                                 fun=self.__read_output_details,
                                 token=LPJmLToken.READ_OUTPUT_SIZE)

        # read and check LPJmL GET_STATUS, send COPAN_OK or COPAN_ERR
        self.__communicate_status()

        # Read all static non time dependent outputs
        self.__grid = np.zeros(shape=(self.__ncell, 2),
                               dtype=self.__output_types[self.__grid_id].type)
        self.__grid[:] = np.nan

        meta_data = self.__read_meta_output(output_id=self.__grid_id)

        self.__grid = LPJmLData(
            data=self.__grid,
            dims=('cell', 'coord'),
            coords=dict(
                cell=np.arange(self.__config.startgrid,
                               self.__config.endgrid+1),
                coord=['longitude', 'latitude']
            ),
            name="grid"
        )
        self.__grid.add_meta(meta_data)

        # Read static outputs
        self.__iterate_operation(
            length=self.__noutput_static, fun=self.__read_static_data,
            token=LPJmLToken.READ_OUTPUT
        )
        # Subtract static inputs from the ones that are read within simulation
        self.__noutput_sim -= self.__noutput_static
        # Get input indices
        input_sockets = self.__config.get_input_sockets()
        self.__input_ids = {
            input_sockets[inp][
                "id"
            ]: inp
            for inp in input_sockets if inp in LPJmLInputType.__members__
        }
        # Get output indices
        output_sockets = self.__config.get_output_sockets()
        self.__output_ids = {
            output_sockets[inp]["index"]: inp for inp in output_sockets
        }
        self.__output_templates = {
            output_sockets[inp]["index"]: self._create_xarray_template(
                output_sockets[inp]["index"]
            ) for inp in output_sockets
        }

    @property
    def config(self):
        """Get the underlyig LpjmLConfig object
        :getter: LpjmLConfig object
        :type: LpjmLConfig
        """
        return self.__config

    @property
    def ncell(self):
        """Get the number of LPJmL cells for in- and output
        :getter: Number of LPJmL cells
        :type: int
        """
        return self.__ncell

    @property
    def grid(self):
        """Get longitude and latitude for every (amount ncell) gridcell as a
        numpy array.
        :getter: Get grid information longitude, lattitude for ncell
        :type: numpy.array
        """
        return self.__grid

    @property
    def operations_left(self):
        operations = []
        if self.__sim_year >= self.__config.start_coupling:
            if self.__sim_year != self.__year_send_input:
                operations.append(LPJmLToken.SEND_INPUT)
        if self.__sim_year >= self.__config.outputyear:
            if self.__sim_year != self.__year_read_output:
                operations.append(LPJmLToken.READ_OUTPUT)
        return operations

    @property
    def sim_years(self):
        """Get a list of all simulation years
        :getter: List of all simulation years
        :type: list
        """
        return [year for year in self.get_sim_years()]

    @property
    def historic_years(self):
        """Get a list of all historic years
        :getter: List of all historic years
        :type: list
        """
        return [year for year in self.get_historic_years(match_period=False)]

    @property
    def coupled_years(self):
        """Get a list of all coupled years
        :getter: List of all coupled years
        :type: list
        """
        return [year for year in self.get_coupled_years(
            match_period=False
        )]

    def get_sim_years(self):
        """Get a generator for all simulation years
        :return: Generator for all simulation years
        :rtype: generator
        """
        start_year = self.__sim_year
        end_year = self.__config.lastyear
        current_year = start_year
        while current_year <= end_year:
            yield current_year
            current_year += 1

    def get_historic_years(self, match_period=True):
        """Get a generator for all historic years
        :return: Generator for all historic years
        :rtype: generator
        """
        start_year = self.__sim_year
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
        :return: Generator for all coupled years
        :rtype: generator
        """
        start_year = self.__sim_year
        end_year = self.config.lastyear
        if match_period and (
            start_year < self.config.start_coupling
        ):
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
        :param id: If True, return cell ids, else return cell indices
        :type id: bool
        :return: Generator for all cells
        :rtype: generator
        """
        start_cell = self.__config.startgrid
        end_cell = self.__config.endgrid
        if id:
            current_cell = start_cell
        else:
            current_cell = 0
            end_cell -= start_cell
        while current_cell <= end_cell:
            yield current_cell
            current_cell += 1

    def read_historic_output(self, to_xarray=True):
        """Read historic output from LPJmL
        :return: Dictionary with output keys and corresponding output as numpy
            arrays with dimensions (ncell, nband)
        :rtype: dict
        """
        # read all historic outputs
        hist_years = list()
        for year in self.get_historic_years():
            hist_years.append(year)
            if year == self.__config.outputyear:
                output_dict = self.read_output(year=year, to_xarray=False)
            elif(year > self.__config.outputyear):
                output_dict = append_to_dict(output_dict,
                                             self.read_output(year=year,
                                                              to_xarray=False))

        for key in output_dict:
            if key in output_dict:
                index = [
                    item[0] for item in self.__output_ids.items()
                    if item[1] == key
                ][0]
                lpjml_output = self._create_xarray_template(
                    index,
                    time_length=len(hist_years)
                )
                # read output values
                if self.__config.output_metafile:
                    meta_output = self.__read_meta_output(index=index)
                else:
                    meta_output = None

                if meta_output:
                    # add band names to output
                    lpjml_output.coords['band'] = meta_output.band_names
                    lpjml_output = lpjml_output.rename(
                        band=f"band ({self.__output_ids[index]})"
                    )
                    # add meta data to output
                    lpjml_output.add_meta(meta_output)

                lpjml_output.coords['time'] = pd.date_range(
                    start=str(hist_years[0]),
                    end=str(hist_years[-1]+1),
                    freq='A'
                )
                lpjml_output.data = output_dict[key]
                output_dict[key] = lpjml_output

        if to_xarray:
            return LPJmLDataSet(output_dict)
        else:
            return output_dict

    def close(self):
        """Close socket channel
        """
        print("Socket channel has been closed.")
        self.__channel.close()

    def send_input(self, input_dict, year):
        """Send input data of iterated year as dictionary to LPJmL. Dictionary
        has to supplied in the form of (example):
        my_dict = {
            'landuse': <numpy_array_with_shape_(ncell, nband)>,
            'with_tillage': <numpy_array_with_shape_(ncell, nband)>
        }
        The dictionary keys must have the same name as input keys supplied to
        the configuration.

        :param input_dict: dict of input keys and corresponding input in the
            form of numpy arrays with dimensions (ncell, nband) (see example
            above)
        :type input_dict: dict
        :param year: supply year for validation
        :type year: int
        """
        # iteration step check - if number of iterations left exceed simulation
        #   steps (analogous to years left)
        # Year check - if procvided year matches internal simulation year
        if year != self.__sim_year:
            raise ValueError(
                f"Year {year} not matches simulated year {self.__sim_year}"
            )
        operations = self.operations_left

        # Check if read_output operation valid
        if not operations or LPJmLToken.SEND_INPUT not in operations:
            raise IndexError(
                f"No send_input operation left for year {year}"
            )
        elif operations.pop(0) != LPJmLToken.SEND_INPUT:
            raise IndexError(f"Invalid operation order. Expected read_output")

        # iterate over outputs for private send_input_data
        self.__iterate_operation(length=self.__ninput,
                                 fun=self.__send_input_data,
                                 token=LPJmLToken.SEND_INPUT,
                                 args={"data": input_dict,
                                       "validate_year": year})

        self.__year_send_input = year

        # check if all operations have been performed and increase sim_year
        if not self.operations_left:
            self.__sim_year += 1

    def read_output(self, year, to_xarray=True):
        """Read LPJmL output data of iterated year. Returned output comes in
        the same format as input is supplied to send_input, with output id/name
        as dict keys:
        outputs = {
            'pft_harvestc': <numpy_array_with_shape_(ncell, nband)>,
            'soilc': <numpy_array_with_shape_(ncell, nband)>
        }

        :param year: supply year for validation
        :type year: int
        :param to_xarray: if True, output is returned as xarray.DataArray
        :type to_xarray: bool
        :return: Dictionary with output id/name as keys and outputs in the form
            of numpy.array
        :rtype: dict
        """
        # Year check - if procvided year matches internal simulation year
        if year != self.__sim_year:
            raise ValueError(
                f"Year {year} not matches simulated year {self.__sim_year}"
            )
        operations = self.operations_left

        # Check if read_output operation valid
        if not operations or LPJmLToken.READ_OUTPUT not in operations:
            raise IndexError(
                f"No read_output operation left for year {year}"
            )
        elif operations.pop(0) != LPJmLToken.READ_OUTPUT:
            raise IndexError(f"Invalid operation order. Expected send_input")

        # iterate over outputs for private read_output_data
        lpjml_output = self.__iterate_operation(length=self.__noutput_sim,
                                                fun=self.__read_output_data,
                                                token=LPJmLToken.READ_OUTPUT,
                                                args={"validate_year": year,
                                                      "to_xarray": to_xarray},
                                                appendix=True)
        if to_xarray:
            lpjml_output = LPJmLDataSet(lpjml_output)

        self.__year_read_output = year

        # check if all operations have been performed and increase sim_year
        if not self.operations_left:
            self.__sim_year += 1

        return lpjml_output

    def read_input(self,
                   start_year=None,
                   end_year=None,
                   copy=True):
        """Read coupled input data from netcdf files and copy them to the
        simulation directory if copy=True. If no start_year and
        end_year are provided, the default start_year and end_year from the
        config are used.
        :param start_year: start year of input data
        :type start_year: int
        :param end_year: end year of input data
        :type end_year: int
        :param copy: if True, input data is copied to simulation directory
        :type copy: bool
        :return: LPJmLDataSet with input data with input names as keys
        :rtype: LPJmLDataSet
        """
        if copy:
            self._copy_input(start_year=start_year,
                             end_year=end_year)
        # read coupled input data from netcdf files (as xarray.DataArray)
        inputs = {key: read_data(
            f"{self.__config.sim_path}/input/{key}.nc",
            var_name=key
        ) for key in self.config.get_input_sockets(id_only=True)}

        inputs = LPJmLDataSet(inputs)
        # define longitide and latitude DataArray (workaround to reduce dims to
        #   cells)
        lons = xr.DataArray(self.grid[:, 0], dims="cell")
        lats = xr.DataArray(self.grid[:, 1], dims="cell")

        other_dim = [
            dim for dim in inputs.dims
            if dim not in ["time", "longitude", "latitude"]
        ]
        if other_dim:
            inputs = inputs.rename_dims({other_dim[0]: 'band'})

        if start_year and end_year:
            kwargs = {"time": [
                year for year in range(start_year, end_year + 1)
            ]}
        elif start_year and not end_year:
            kwargs = {"time": [
                year for year in range(start_year,
                                       max(inputs.time.values)+1)
            ]}
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
        # create same format as before but with selected numpy arrays instead
        # of xarray.DataArray
        inputs = inputs.sel(
            longitude=lons, latitude=lats, method="nearest", **kwargs
        ).transpose("cell", ..., "time")

        return inputs

    def _copy_input(self,
                    start_year,
                    end_year):
        """Copy and convert and save input files as NetCDF4 files to input
        directory for selected years to make them easily readable as well as to
        avoid large file sizes.
        """
        input_path = f"{self.__config.sim_path}/input"
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

        # iterate over each inputs to be send via sockets (get initial values)
        for key in sock_inputs:
            # check if working on the cluster (workaround by Ciaron)
            #   (might be adjusted to the new cluster coming soon ...)
            if self.config.inpath and (
                not sock_inputs[key]['name'].startswith("/")
            ):
                sock_inputs[key]['name'] = (
                    f"{self.config.inpath}/{sock_inputs[key]['name']}"
                )
            # get input file name
            file_name_clm = sock_inputs[key]['name'].split("/")[-1]
            # name tmp file after original name (even though could be random)
            file_name_tmp = (
                f"{file_name_clm.split('.')[0]}_tmp.clm"
            )
            # predefine cut clm command for reusage
            cut_clm_start = [f"{self.__config.model_path}/bin/cutclm",
                             str(start_year),
                             sock_inputs[key]['name'],
                             f"{temp_dir}/1_{file_name_tmp}"]

            start_year_check = start_year
            # run cut clm file before start year and after end year in sequence
            while True:
                try:
                    run(cut_clm_start, stdout=open(os.devnull, 'wb'),
                        check=True)
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
            cut_clm_end = [f"{self.__config.model_path}/bin/cutclm",
                           "-end", str(end_year),
                           f"{temp_dir}/1_{file_name_tmp}",
                           f"{temp_dir}/2_{file_name_tmp}"]
            run(cut_clm_end, stdout=open(os.devnull, 'wb'))
            # a flag for multi (categorical) band input - if true, set
            #   "-landuse"
            if getattr(LPJmLInputType, key).bands:
                is_multiband = "-landuse"
            else:
                is_multiband = None
            # a flag for integer input - if true, set "-int"
            if getattr(LPJmLInputType, key).type == int:
                is_int = "-intnetcdf"
            else:
                is_int = None
            # default grid file (only valid for 0.5 degree inputs)
            if self.config.input.coord.name.startswith("/"):
                grid_file = self.config.input.coord.name
            else:
                grid_file = (
                    f"{self.config.inpath}/{self.config.input.coord.name}"
                )
            # convert clm input to netcdf files
            conversion_cmd = [
                f"{self.__config.model_path}/bin/clm2cdf", is_int,
                is_multiband, key,
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

    def __iterate_operation(self, length, fun, token, args=None,
                            appendix=False):
        """Iterate reading/sending operation for sequence of inputs and/or
        outputs
        """
        # check if read token matches expected token and return read token
        token_check, received_token = self.__check_token(token)
        if not token_check:
            self.close()
            raise ValueError(
                f"Received LPJmLToken {received_token.name} is not {token.name}"
            )
        # execute method on channel and if supplied further method arguments
        if not args:
            result = fun()
        else:
            result = fun(**args)
        # recursive iteration
        if length > 1:
            # if appendix results are appended/extended and returned as list
            if appendix:
                result = {**result, **self.__iterate_operation(
                    length-1, fun, token, args, appendix)
                }
                return result
            else:
                self.__iterate_operation(length-1, fun, token, args)
        else:
            if appendix:
                return result

    def __check_token(self, token):
        """ check if read token matches the expected token
        """
        received_token = read_token(self.__channel)
        if received_token == token:
            return True, received_token
        else:
            return False, received_token

    def __send_band_size(self):
        """Send input band size for read index to socket
        """
        index = read_int(self.__channel)
        # convert received LPJmL data types into Python compatible types
        self.__set_input_types(index)
        if index in self.__input_bands.keys():
            # Send number of bands
            send_int(self.__channel, val=self.__input_bands[index])
        else:
            self.close()
            raise ValueError(f"Input of input ID {index} not supported.")

    def __set_input_types(self, index):
        """Convert received LPJmL data types into Python compatible types
        """
        self.__input_types[index] = LPJmlValueType(
            read_int(self.__channel))

    def __get_config_input_sockets(self):
        """Get and validate input sockets, check if defined in LPJmLInputType
        class. If not add to LPJmLInputType.
        """
        # get defined input sockets
        sockets = self.__config.get_input_sockets()
        # filter input names
        input_names = [inp.name for inp in LPJmLInputType]
        # check if input is defined in LPJmLInputType (band size required)
        valid_inputs = {
            getattr(LPJmLInputType, sock).value: getattr(
                LPJmLInputType, sock
            ).nband for sock in sockets if sock in input_names
        }
        if len(sockets) != len(valid_inputs):
            self.close()
            raise ValueError(
                f"Configurated sockets {list(sockets.keys())} not defined in" +
                f" {input_names}!"
            )
        return valid_inputs

    def __read_output_details(self):
        """Read output details per output index (timesteps, number of bands,
        data types) from socket
        """
        index = read_int(self.__channel)
        # Get number of steps for output
        self.__output_steps[index] = read_int(self.__channel)
        # Get number of bands for output
        self.__output_bands[index] = read_int(self.__channel)
        # Get datatype for output
        self.__output_types[index] = LPJmlValueType(
            read_int(self.__channel))
        # Check for static output if so increment static output counter
        if index == self.__globalflux_id:
            pass
        elif index in self.__static_id:
            self.__noutput_static += 1
        # check if only annual timesteps were set
        if self.__output_steps[index] > 1:
            send_int(self.__channel, CopanStatus.COPAN_ERR.value)
            self.close()
            raise ValueError(f"Time step {self.__output_steps[index]} " +
                             f" for output ID {index} invalid.")
        else:
            send_int(self.__channel, CopanStatus.COPAN_OK.value)

    def __communicate_status(self):
        """Check if LPJmL token equals GET_STATUS, send OK or ERROR
        """
        check_token = LPJmLToken(read_int(self.__channel))
        if check_token == LPJmLToken.GET_STATUS:
            if self.__ninput != 0 and self.__noutput != 0:
                send_int(self.__channel, CopanStatus.COPAN_OK.value)
            else:
                self.close()
                send_int(self.__channel, CopanStatus.COPAN_ERR.value)
                raise ValueError("No inputs OR outputs defined.")
        else:
            raise ValueError(f"Got LPJmLToken {check_token.name}, though " +
                             f"LPJmLToken {LPJmLToken.GET_STATUS} expected.")

    def __read_static_data(self):
        """Read static data to be called within initialization of coupler.
        Currently only grid data supported
        """
        index = read_int(self.__channel)
        if index == self.__grid_id:
            # Check for datatype grid data and assign right read function and
            #   define scale factor
            if self.__output_types[self.__grid_id].type == int:
                read_grid_val = read_short
                type_fact = 0.01
            else:
                read_grid_val = read_float
                type_fact = 1
            # iterate over ncell and read + assign lon and lat values
            for ii in range(0, self.__ncell):
                self.__grid[ii, 0] = (
                    read_grid_val(self.__channel) * type_fact
                )
                self.__grid[ii, 1] = (
                    read_grid_val(self.__channel) * type_fact
                )
            self.__grid.coords['longitude'] = (
                ('cell',), self.__grid.data[:, 0]
            )
            self.__grid.coords['latitude'] = (
                ('cell',), self.__grid.data[:, 1]
            )

    def _create_xarray_template(self, index, time_length=1):
        """Create xarray template for output data
        """
        bands = self.__output_bands[index]

        # create output numpy array template to be filled with output
        output_tmpl = np.zeros(
            shape=(self.__ncell, bands, time_length),  # time = 1
            dtype=self.__output_types[index].type)

        # Check if data array is of type integer, use -9999 for nan
        if self.__output_types[index].type == int:
            output_tmpl[:] = -9999
        else:
            output_tmpl[:] = np.nan

        # create xarray data array with correct dimensions and coord
        output_tmpl = LPJmLData(
            data=output_tmpl,
            dims=('cell', 'band', 'time'),
            coords=dict(
                cell=np.arange(self.__config.startgrid,
                               self.__config.endgrid+1),
                longitude=(
                    ['cell'], self.__grid.coords['longitude'].values
                ),
                latitude=(
                    ['cell'], self.__grid.coords['latitude'].values
                ),
                band=np.arange(bands),  # [str(i) for i in range(bands)],
                time=np.arange(time_length)
            ),
            name=self.__output_ids[index],
        )
        return output_tmpl

    def __send_input_data(self, data, validate_year):
        """Send input data checks supplied object type, object dimensions data
        format and year for input index. If set correct executes private
        send_input_values method that does the sending.
        """
        index = read_int(self.__channel)
        if isinstance(data, LPJmLDataSet):
            data = data.to_numpy()
        elif not isinstance(data[self.__input_ids[index]], np.ndarray):
            self.close()
            raise TypeError("Unsupported object type. Please supply a numpy " +
                            "array with the dimension of (ncells, nband).")

        # type check conversion
        if self.__input_types[index].type == float:
            type_check = np.floating
        elif self.__input_types[index].type == int:
            type_check = np.integer

        if not np.issubdtype(
            data[self.__input_ids[index]].dtype, type_check
        ):
            self.close()
            raise TypeError(
                f"Unsupported type: {data[self.__input_ids[index]].dtype} " +
                "Please supply a numpy array with data type: " +
                f"{np.dtype(self.__input_types[index].type)}."
            )
        year = read_int(self.__channel)
        if not validate_year == year:
            self.close()
            raise ValueError(f"The expected year: {validate_year} does not " +
                             f"match the received year: {year}")
        if index in self.__input_ids.keys():
            # get corresponding number of bands from LPJmLInputType class
            bands = LPJmLInputType(index).nband
            if not np.shape(
                data[self.__input_ids[index]]
            ) == (self.__ncell, bands):
                if bands == 1 and not np.shape(
                    data[self.__input_ids[index]]
                ) == (self.__ncell, ):
                    self.close()
                    raise ValueError(
                        "The dimensions of the supplied data: " +
                        f"{(self.__ncell, bands)} does not match the " +
                        f"needed dimensions for {self.__input_ids[index]}" +
                        f": {(self.__ncell, bands)}."
                    )
            # execute sending values method to actually send the input to
            #   socket
            self.__send_input_values(data[self.__input_ids[index]])

    def __send_input_values(self, data):
        """Iterate over all values to be send to socket. Recursive iteration
        with correct order of bands and cells for inputs
        """
        dims = list(np.shape(data))
        if len(dims) == 1:
            dims.append(1)
            one_band = True
        else:
            one_band = False

        if "int" in str(data.dtype):
            send_function = send_int
        else:
            send_function = send_float

        dims[0] -= 1
        dims[1] -= 1
        cells = dims[0]
        bands = dims[1]
        # iterate over bands (first) and cells (second)
        while cells >= 0 and bands >= 0:
            # send float value for input (are all inputs floats?) - indices via
            #   decremented cells, bands and orignal dims
            if one_band:
                send_function(self.__channel, data[dims[0]-cells])
            else:
                send_function(
                    self.__channel, data[dims[0]-cells, dims[1]-bands]
                )

            if bands > 0:
                bands -= 1
            elif bands == 0 and cells >= 0:
                cells -= 1
                bands = dims[1]

    def __read_output_data(self, validate_year, to_xarray=True):
        """Read output data checks supplied year and sets numpy array template
        for corresponding output (index). If set correct executes
        private read_output_values method to read the corresponding output.
        """
        index = read_int(self.__channel)
        year = read_int(self.__channel)
        if not validate_year == year:
            self.close()
            raise ValueError(f"The expected year: {validate_year} does not " +
                             f"match the received year: {year}")
        if index in self.__output_ids.keys():
            if not to_xarray:
                # get corresponding number of bands
                bands = self.__output_bands[index]
                # create output numpy array template to be filled with output
                output_tmpl = np.zeros(
                    shape=(self.__ncell, bands, 1),  # time = 1
                    dtype=self.__output_types[index].type)
                # Check if data array is of type integer, use -9999 for nan
                if self.__output_types[index].type == int:
                    output_tmpl[:] = -9999
                else:
                    output_tmpl[:] = np.nan
            else:
                # get corresponding number of bands
                output_tmpl = self.__output_templates[index]
                # read output values
                if self.__config.output_metafile:
                    meta_output = self.__read_meta_output(index=index)
                else:
                    meta_output = None

                if meta_output:
                    # add band names to output
                    output_tmpl.coords['band'] = meta_output.band_names
                    output_tmpl = output_tmpl.rename(
                        band=f"band ({self.__output_ids[index]})"
                    )
                    # add meta data to output
                    output_tmpl.add_meta(meta_output)

                output_tmpl.coords['time'] = pd.date_range(
                    str(year), periods=1, freq='A'
                )

            # read and assign corresponding values from socket to numpy array
            output = self.__read_output_values(
                output=output_tmpl,
                dims=list(
                    np.shape(output_tmpl)),
                lpjml_type=self.__output_types[index]
            )
            # as list for appending/extending as list
            return {self.__output_ids[index]: output}

    def __read_output_values(self,
                             output,
                             dims=None,
                             lpjml_type=LPJmlValueType(3)):
        """Iterate over all values to be read from socket. Recursive iteration
        with correct order of cells and bands for outputs
        """
        dims[0] -= 1
        dims[1] -= 1
        cells = dims[0]
        bands = dims[1]

        if dims[0] > 0 and dims[1] == 0:
            one_band = True
        else:
            one_band = False

        # iterate over cells (first) and bands (second)
        while cells >= 0 and bands >= 0:
            # read float value for output (are all outputs floats?) - indices
            #   via decremented cells, bands and orignal dims
            if one_band:
                output[dims[0]-cells] = lpjml_type.read_fun(self.__channel)
            else:
                output[dims[0]-cells, dims[0]-bands] = lpjml_type.read_fun(
                    self.__channel
                )

            if cells > 0:
                cells -= 1
            elif cells == 0 and bands >= 0:
                bands -= 1
                cells = dims[0]
        return output

    def __read_meta_output(self, index=None, output_id=None):
        """Read meta output data from socket. Returns dictionary with
        corresponding meta output id and value.
        """
        if output_id is not None:
            output = self.__config.output[output_id]
        elif index is not None:
            output = [
                out for out in self.__config.output
                if out.id == self.__output_ids[index]
            ][0]
        else:
            raise ValueError("Either index or output_id must be supplied")

        # Read meta data from of corresponding output id using config info
        lpjml_meta = read_meta(f"{output.file.name}.json")

        # Set meta data correct for coupling format
        lpjml_meta.nstep = 1
        lpjml_meta.timestep = 1
        lpjml_meta.format = "sock"

        # Remove unnecessary attributes
        delattr(lpjml_meta, "bigendian")
        delattr(lpjml_meta, "filename")

        return lpjml_meta

    def __repr__(self):
        """Representation of the Coupler object
        """
        port = self.__channel.getsockname()[1]
        summary = f"<pycoupler.{self.__class__.__name__}>"
        summary = "\n".join([
            summary,
            f"Simulation:  (version: {self.version}, localhost:{port})",
            f"  * sim_year   {self.__sim_year}",
            f"  * ncell      {self.__ncell}",
            f"  * ninput     {self.__ninput}",
        ])
        summary = "\n".join([summary, self.__config.__repr__(sub_repr=True)])

        return summary
