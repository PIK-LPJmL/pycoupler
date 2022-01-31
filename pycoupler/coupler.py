import socket
import struct
import numpy as np
from enum import Enum, auto
from operator import itemgetter
from config import read_config


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


def read_token(channel):
    """read integer as Token (Enum class)
    """
    # create Token object
    return Token(read_int(channel))


class Token(Enum):
    """Available tokens"""
    GET_DATA: int = 0       # Receiving data from COPAN
    PUT_DATA: int = 1       # Sending data to COPAN
    GET_DATA_SIZE: int = 2  # Receiving data size from COPAN
    PUT_DATA_SIZE: int = 3  # Sending data size to COPAN
    END_DATA: int = 4       # Ending communication
    GET_STATUS: int = 5     # Check status of COPAN


class CopanStatus(Enum):
    """Available Inputs"""
    COPAN_OK: int = 0
    COPAN_ERR: int = -1


class Inputs(Enum):
    """Available Inputs"""
    landuse: int = 6  # number of bands in landuse data
    fertilizer_nr: int = 18  # number of bands in fertilizer data
    manure_nr: int = 19  # number of bands in manure data
    residue_on_field: int = 8  # number of bands in residue data
    with_tillage: int = 7  # number of bands in tillage data

    @property
    def band(self):
        if self.name == "landuse":
            return 64
        elif self.name in ["fertilizer_nr", "manure_nr", "residue_on_field"]:
            return 32
        elif self.name == "with_tillage":
            return 2


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


class Coupler:
    """ Coupler class serves as the interface between LPJmL and the model to
    couple with. It initiates a socket channel, provides it and follows the
    initialization protocol (->: receiving, <-: sending, for: iteration):
    -> version (int32)
    -> number of cells (int32)
    -> number of input streams (int32)
    -> number of output streams (int32)
    for <number of input streams>:
        -> GET_DATA_SIZE (int32)
        -> index (int32)
        -> type(int32)
        <- number of bands (int32)
    for <number of output streams>:
        -> PUT_DATA_SIZE (int32)
        -> index (int32)
        -> number of steps(int32)
        -> number of bands (int32)
        -> type(int32)
    for <number of static outputs (config)>:
        -> PUT_DATA (int_32)
        -> index (int32)
        if <static output == grid>:
            -> data(cell1, band1), …, data(celln, band1)
               data(cell1, band2), …, data(celln, band2)
        else:
            <next>
    The protocol may vary from version to version hence this only reflects the
    current state of the protocol. Please use the Coupler Instance throughout
    the Simulation and do not close or reinitiate in between.

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
        self.__niteration = len(range(self.__config.firstyear,
                                      self.__config.lastyear+1))
        self.__output_niteration = self.__input_niteration = self.__niteration
        # open/initialize socket channel
        self.__channel = opentdt(host, port)
        # Check coupler protocol version
        self.version = read_int(self.__channel)
        if self.version != version:
            self.close_channel()
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
        self.__iterate_operation(
            length=self.__ninput, fun=self.__send_band_size,
            token=Token.GET_DATA_SIZE,
        )
        # init list to be filled with nbands, nsteps per output
        self.__output_bands = [-1] * len(self.config.outputvar)
        self.__output_steps = [-1] * len(self.config.outputvar)
        self.__output_types = [-1] * len(self.config.outputvar)
        # init counter to be filled with number of static outputs
        self.__noutput_static = 0
        # Check for static output
        outputs_avail = self.__config.get_outputs_avail(id_only=False)
        # get s
        self.__globalflux_id = [out["id"] for out in outputs_avail if out[
            "name"] == "globalflux"][0]
        self.__grid_id = [out["id"] for out in outputs_avail if out[
            "name"] == "grid"][0]
        self.__static_id = [out["id"] for out in outputs_avail if out[
            "name"] in ["grid", "country", "region"]]
        # Get number of bands per cell for each output data stream
        self.__iterate_operation(length=self.__noutput,
                                 fun=self.__read_output_details,
                                 token=Token.PUT_DATA_SIZE)
        # read and check LPJmL GET_STATUS, send COPAN_OK or COPAN_ERR
        self.__communicate_status()
        # Read all static non time dependent outputs
        self.__grid = np.zeros(shape=(self.__ncell, 2),
                               dtype=self.__output_types[self.__grid_id])
        self.__iterate_operation(
            length=self.__noutput_static, fun=self.__read_static_data,
            token=Token.PUT_DATA
        )
        # subtract static inputs from the ones that are read within simulation
        self.__noutput_sim -= self.__noutput_static
        # get input indices
        input_sockets = self.__config.get_input_sockets()
        self.__input_ids = {
            input_sockets[inp][
                "id"
            ]: inp for inp in input_sockets if inp in Inputs.__members__
        }
        # get output indices
        output_sockets = self.__config.get_output_sockets()
        self.__output_ids = {
            output_sockets[inp]["index"]: inp for inp in output_sockets
        }

    @property
    def config(self):
        """Get the underlyig LpjmLConfig object
        :getter: LpjmLConfig object
        :type: LpjmLConfig
        """
        return self.__config

    @property
    def niteration(self):
        """Get the number of coupled simulation iterations left (!)
        :getter: Number of left coupled iterations
        :type: int
        """
        return self.__niteration

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
        :getter: Get grid information lon, lat for ncell
        :type: numpy.array
        """
        return self.__grid

    def close_channel(self):
        """Close socket channel
        """
        self.__channel.close()

    def send_inputs(self, input_dict, year):
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
        if self.__input_niteration < 0:
            self.close_channel()
            raise IndexError("Number of simulation iterations exceeded")
        # iteration step check - if number of input iterations left does not
        #   match output iterations left (analogous to years left)
        if self.__input_niteration != self.__output_niteration:
            self.close_channel()
            raise RuntimeError("Sequence of send_input and read_output " +
                               "calls not matching.")
        # create list to temporarily store input data from input_dict
        input_list = np.array([None] * self.__ninput)
        input_list[list(self.__input_ids.keys())] = itemgetter(
            *list(self.__input_ids.values()))(input_dict)
        # iterate over outputs for private send_input_data
        self.__iterate_operation(length=self.__ninput,
                                 fun=self.__send_input_data,
                                 token=Token.GET_DATA,
                                 args={"data": input_list,
                                       "validate_year": year})
        # decrement input iterations left (analogous to years left)
        self.__input_niteration -= 1

    def read_outputs(self, year):
        """Read LPJmL output data of iterated year. Returned output comes in
        the same format as input is supplied to send_input, with output id/name
        as dict keys:
        outputs = {
            'pft_harvestc': <numpy_array_with_shape_(ncell, nband)>,
            'soilc': <numpy_array_with_shape_(ncell, nband)>
        }

        :param year: supply year for validation
        :type year: int
        :return: Dictionary with output id/name as keys and outputs in the form
            of numpy.array
        :rtype: dict
        """
        # iteration step check - if number of iterations left exceed simulation
        #   steps (analogous to years left)
        if self.__output_niteration < 0:
            self.close_channel()
            raise IndexError("Number of simulation iterations exceeded")
        # iteration step check - if number of output iterations left does not
        #   match input iterations left (analogous to years left)
        if self.__input_niteration + 1 != self.__output_niteration:
            self.close_channel()
            raise RuntimeError("Sequence of send_input and read_output " +
                               "calls not matching.")
        # iterate over outputs for private send_output_data
        output_list = self.__iterate_operation(length=self.__noutput_sim,
                                               fun=self.__read_output_data,
                                               token=Token.PUT_DATA,
                                               args={"validate_year": year})
        # convert output_list again to output_dict (format as above) and return
        output_dict = {self.__output_ids[idx]: output_list[
            idx] for idx in self.__output_ids.keys()}
        # decrement output iterations left (analogous to years left)
        self.__output_niteration -= 1
        # decrement general iteration counter
        self.__niteration -= 1
        return output_dict

    def __iterate_operation(self, length, fun, token, args=None,
                            appendix=False):
        """Iterate reading/sending operation for sequence of inputs and/or
        outputs
        """
        # check if read token matches expected token and return read token
        token_check, received_token = self.__check_token(token)
        if not token_check:
            self.close_channel()
            raise ValueError(
                f"Token {received_token.name} is not {token.name}"
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
                result.extend(self.__iterate_operation(length-1, fun, token,
                                                       args, appendix))
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
            return True, token
        else:
            return False, token

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
            self.close_channel()
            raise ValueError(f"Input of input ID {index} not supported.")

    def __set_input_types(self, index):
        """Convert received LPJmL data types into Python compatible types
        """
        self.__input_types[index] = LpjmlTypes(
            read_int(self.__channel)).type

    def __get_config_input_sockets(self):
        """Get and validate input sockets, check if defined in Inputs Class. If
        not add to Inputs.
        """
        # get defined input sockets
        sockets = self.__config.get_input_sockets()
        # filter input names
        input_names = [inp.name for inp in Inputs]
        # check if input is defined in Inputs (band size required)
        valid_inputs = {getattr(Inputs, sock).value: getattr(
            Inputs, sock).band for sock in sockets if sock in input_names}
        if len(sockets) != len(valid_inputs):
            self.close_channel()
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
        self.__output_types[index] = LpjmlTypes(
            read_int(self.__channel)).type
        # Check for static output if so increment static output counter
        if index == self.__globalflux_id:
            self.flux = [
                self.__output_types[index](0)] * self.__output_bands[index]
        elif index in self.__static_id:
            self.__noutput_static += 1
        # check if only annual timesteps were set
        if self.__output_steps[index] > 1:
            send_int(self.__channel, CopanStatus.COPAN_ERR.value)
            self.close_channel()
            raise ValueError(f"Time step {self.__output_steps[index]} " +
                             f" for output ID {index} invalid.")
        else:
            send_int(self.__channel, CopanStatus.COPAN_OK.value)

    def __communicate_status(self):
        """Check if LPJmL token equals GET_STATUS, send OK or ERROR
        """
        check_token = Token(read_int(self.__channel))
        if check_token == Token.GET_STATUS:
            if self.__ninput != 0 and self.__noutput != 0:
                send_int(self.__channel, CopanStatus.COPAN_OK.value)
            else:
                self.close_channel()
                send_int(self.__channel, CopanStatus.COPAN_ERR.value)
                raise ValueError("No inputs OR outputs defined.")
        else:
            raise ValueError(f"Got Token {check_token.name}, though " +
                             f"Token {Token.GET_STATUS} was expected.")

    def __read_static_data(self):
        """Read static data to be called within initialization of coupler.
        Currently only grid data supported
        """
        index = read_int(self.__channel)
        if index == self.__grid_id:
            # Check for datatype grid data and assign right read function and
            #   define scale factor
            if self.__output_types[self.__grid_id] == int:
                read_grid_val = read_short
                type_fact = 0.01
            else:
                read_grid_val = read_float
                type_fact = 1
            # iterate over ncell and read + assign lon and lat values
            for ii in range(0, self.__ncell):
                self.__grid[ii, 0] = read_grid_val(self.__channel) * type_fact
                self.__grid[ii, 1] = read_grid_val(self.__channel) * type_fact

    def __send_input_data(self, data, validate_year):
        """Send input data checks supplied object type, object dimensions data
        format and year for input index. If set correct executes private
        send_input_values method that does the sending.
        """
        if not isinstance(data, np.ndarray):
            self.close_channel()
            raise TypeError("Unsupported object type. Please supply a numpy " +
                            "array with the dimension of (ncells, nband).")
        index = read_int(self.__channel)
        if not isinstance(data, self.__input_types[index]):
            self.close_channel()
            raise TypeError(
                f"Unsupported data type: {data.dtype} " +
                "Please supply a numpy array with data type: " +
                f"{self.__input_types[index]}."
            )
        year = read_int(self.__channel)
        if not validate_year == year:
            self.close_channel()
            raise ValueError(f"The expected year: {validate_year} does not " +
                             f"match the received year: {year}")
        if index in self.__input_ids.keys():
            # get corresponding number of bands from Inputs class
            bands = Inputs(index).band
            if not np.shape(data) == (self.__ncell, bands):
                self.close_channel()
                ValueError(
                    "The dimensions of the supplied data: " +
                    f"{(self.__ncell, bands)} does not match the required " +
                    f"dimensions for {self.__input_ids[index]}: " +
                    f"{(self.__ncell, bands)}."
                )
            # execute sending values method to actually send the input to
            #   socket
            self.__send_input_values(data)

    def __send_input_values(self, data, cells=None, bands=None, dims=None):
        """Iterate over all values to be send to socket. Recursive iteration
        with correct order of bands and cells for inputs
        """
        # initiate cells, bands and dims for first level call
        if not cells and not bands:
            dims = np.shape(data)
            cells = dims[0]
            bands = dims[1]
        # send float value for input (are all inputs floats?) - indices via
        #   decremented cells, bands and orignal dims
        send_float(self.__channel, data[dims[0]-cells, dims[1]-bands])
        # iterate over bands first
        if bands != 0 and cells != 0:
            self.__send_input_values(data=data,
                                     cells=cells,
                                     bands=bands-1,
                                     dims=dims)
        # iterate over cells second
        elif bands == 0 and cells != 0:
            self.__send_input_values(data=data,
                                     cells=cells-1,
                                     bands=dims[1],
                                     dims=dims)

    def __read_output_data(self, validate_year):
        """Read output data checks supplied year and sets numpy array template
        for corresponding output (index). If set correct executes
        private read_output_values method to read the corresponding output.
        """
        index = read_int(self.__channel)
        year = read_int(self.__channel)
        if not validate_year == year:
            self.close_channel()
            raise ValueError(f"The expected year: {validate_year} does not " +
                             f"match the received year: {year}")
        if index in self.__output_ids.keys():
            # get corresponding number of bands
            bands = self.__output_bands[index]
            # create output numpy array template to be filled with output
            output_tmpl = np.zeros(shape=(self.__ncell, bands),
                                   dtype=self.__output_types[index])
            # read and assign corresponding values from socket to numpy array
            output = self.__read_output_values(output=output_tmpl,
                                               cells=self.__ncell, bands=bands,
                                               dims=np.shape(output_tmpl))
            # as list for appending/extending as list
            return [output]
        else:
            # also as list even if empty/None
            return [None]

    def __read_output_values(self, output, cells=None, bands=None, dims=None):
        """Iterate over all values to be read from socket. Recursive iteration
        with correct order of cells and bands for outputs
        """
        # read float value for output (are all outputs floats?) - indices via
        #   decremented cells, bands and orignal dims
        output[dims[0]-cells, dims[0]-bands] = read_float(self.__channel)
        # iterate over cells first
        if bands != 0 and cells != 0:
            output = self.__read_output_values(output=output,
                                               cells=cells-1,
                                               bands=bands,
                                               dims=dims)
        # iterate over bands second
        elif bands != 0 and cells == 0:
            output = self.__read_output_values(output=output,
                                               cells=dims[0],
                                               bands=bands-1,
                                               dims=dims)
        return output
