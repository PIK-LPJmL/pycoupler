import socket
import struct
import numpy as np
from enum import Enum
from operator import itemgetter
from config import read_config


def recvall(channel, size):
    string = ""
    nbytes = 0
    while bytes < size:
        string += channel.recv(size-nbytes)
        nbytes += len(string)
    return str


def write_int(channel, val):
    channel.sendall(struct.pack('i', val))


def read_int(channel):
    intstr = recvall(channel, struct.calcsize('i'))
    inttup = struct.unpack('i', intstr)
    return inttup[0]


def read_short(channel):
    intstr = recvall(channel, struct.calcsize('h'))
    inttup = struct.unpack('h', intstr)
    return inttup[0]


def write_char(channel, val):
    channel.sendall(val)


def read_char(channel):
    # The size of a char is always 1, and there is nothing to unpack
    c = recvall(channel, 1)
    return c


def write_float(channel, val):
    channel.sendall(struct.pack('f', val))


def read_float(channel):
    floatstr = recvall(channel, struct.calcsize('f'))
    floattup = struct.unpack('f', floatstr)
    return floattup[0]


def opentdt(port):
    # create an INET, STREAMing socket
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind the socket to a public host, and a well-known port
    serversocket.bind(("", port))
    # become a server socket
    serversocket.listen(5)
    # accept connections from outside
    (channel, address) = serversocket.accept()
    channel.send('1')
    known_int = read_int(channel)
    num = read_int(channel)
    num = 1
    write_int(channel, num)
    return channel


def read_token(channel):
    # create Token object
    return Token(read_int(channel))


class Token(Enum):
    """Available tokens"""
    GET_DATA: int = 0       # Receiving data from COPAN
    PUT_DATA: int = 1       # Sending data to COPAN
    GET_DATA_SIZE: int = 2  # Receiving data size from COPAN
    PUT_DATA_SIZE: int = 3  # Sending data size to COPAN
    END_DATA: int = 4       # Ending communication


class Inputs(Enum):
    """Available tokens"""
    landuse: int = 64  # number of bands in landuse data
    fertilizer_nr: int = 32  # number of bands in fertilizer data
    manure_nr: int = 32  # number of bands in manure data
    residue_on_field: int = 32  # number of bands in residue data
    with_tillage: int = 2  # number of bands in tillage data


class LpjmlTypes(Enum):
    """Available datatypes
    """
    LPJ_BYTE: int = 0
    LPJ_SHORT: int = 1
    LPJ_INT: int = 2
    LPJ_FLOAT: int = 3
    LPJ_DOUBLE: int = 4

    def to_type(self):
        if self.value > 2:
            return float
        else:
            return int


class Coupler:
    # constructor; set channel
    def __init__(self, config_file, version=2, port=2224):
        self.config = read_config(config_file)
        # open/initialize socket channel
        self.channel = opentdt(port)
        # Check coupler protocol version
        self.version = read_int(self.channel)
        if (self.version != version):
            self.channel.close()
            raise ValueError(
                f"Invalid coupler version {version}, must be {self.version}"
            )
        # read amount of LPJml cells
        self.ncell = read_int(self.channel)
        # read amount of input streams
        self.n_in = read_int(self.channel)
        # read amount of output streams
        self.n_out = read_int(self.channel)
        # Send number of bands per cell for each input data stream
        self.__iterate_operation(
            length=self.n_in, fun=self.__write_input_size,
            token=Token.GET_DATA_SIZE,
            args={"input_bands": self.__read_config_sockets()}
        )
        self.__in_types = [-1] * self.n_in
        # init list to be filled with nband per output
        self.__out_bands = [-1] * self.n_out
        self.__out_steps = [-1] * self.n_out
        self.__out_types = [-1] * self.n_out
        # self.__out_data = [0.0] * self.ncell
        # init counter to be filled with number of static outputs
        self.__out_static = 0
        # Check for static output
        outputs_avail = self.config.get_outputs_avail(id_only=False)
        self.__globalflux_id = [out["id"] for out in outputs_avail if out[
            "name"] == "globalflux"][0]
        self.__grid_id = [out["id"] for out in outputs_avail if out[
            "name"] == "grid"][0]
        self.__static_id = [out["id"] for out in outputs_avail if out[
            "name"] in ["grid", "country", "region"]]
        # Get number of bands per cell for each output data stream
        self.__iterate_operation(length=self.n_out,
                                 fun=self.__read_output_details,
                                 token=Token.PUT_DATA_SIZE)
        # Read all static non time dependent outputs
        self.grid = np.zeros(shape=(self.ncell, 2),
                             dtype=self.__out_types[self.__grid_id])
        self.__iterate_operation(
            length=self.__out_static, fun=self.__read_static_data,
            token=Token.PUT_DATA
        )
        self.n_out -= self.__out_static
        # get input indices
        input_sockets = self.config.get_input_sockets()
        self.__input_ids = {
            input_sockets[inp][
                "id"
            ]: inp for inp in input_sockets if inp in Inputs.__members__
        }
        # get output indices
        output_sockets = self.config.get_output_sockets()
        self.__output_ids = {
            output_sockets[inp]["index"]: inp for inp in output_sockets
        }

    def write_input(self, data_dict, year):
        data_list = np.array([None] * self.n_in)
        data_list[list(self.__input_ids.keys())] = itemgetter(
            *list(self.__input_ids.values()))(data_dict)
        self.__iterate_operation(length=self.n_in,
                                 fun=self.__write_input_data,
                                 token=Token.GET_DATA,
                                 args={"data": data_list,
                                       "validate_year": year})

    def read_output(self, year):
        output_list = self.__iterate_operation(length=self.n_out,
                                               fun=self.__read_output_data,
                                               token=Token.PUT_DATA,
                                               args={"validate_year": year})

        output_dict = {self.__output_ids[idx]: output_list[
            idx] for idx in self.__output_ids.keys()}
        return output_dict

    def __iterate_operation(self, length, fun, token, args=None,
                            appendix=False):
        token_check, received_token = self.__check_token(token)
        if not token_check:
            raise ValueError(
                f"Token {received_token.name} is not {token.name}"
            )
        result = fun(self.channel, **args)
        # recursive iteration
        if length > 0:
            if appendix:
                result.append(self.__iterate_operation(length-1, fun, token,
                                                       args, appendix))
                return result
            else:
                self.__iterate_operation(length-1, fun, token, args)
        else:
            if appendix:
                return result

    def __check_token(self, token):
        received_token = read_token(self.channel)
        if received_token is token:
            return True, token
        else:
            return False, token

    def __write_input_size(self, input_bands):
        index = read_int(self.channel)
        self.__in_types[index] = LpjmlTypes(read_int(self.channel)).to_type()
        if index in input_bands.keys():
            # Send number of bands
            write_int(self.channel, val=input_bands[index])
        else:
            write_int(self.channel, val=0)

    def __read_config_sockets(self):
        sockets = self.config.get_input_sockets()
        input_names = [inp.name for inp in Inputs]
        valid_inputs = {
            sock: getattr(
                Inputs, sock
            ).value for sock in sockets if sock in input_names
        }
        if len(sockets) != len(valid_inputs):
            raise ValueError(
                f"Configurated sockets {sockets.keys()} not defined in " +
                f"{input_names}!"
            )
        return valid_inputs

    def __read_output_details(self):
        index = read_int(self.channel)
        # Get number of steps for output
        self.__out_steps[index] = read_int(self.channel)
        # Get number of bands for output
        self.__out_bands[index] = read_int(self.channel)
        # Get datatype for output
        self.__out_types[index] = LpjmlTypes(read_int(self.channel)).to_type()
        # Check for static output
        if index in self.__globalflux_id:
            self.flux = [self.__out_types[index](0)] * self.__out_bands[index]
        elif index in self.__static_id:
            self.__out_static += 1

    def __read_static_data(self):
        index = read_int(self.channel)
        if index == self.__grid_id:
            if LpjmlTypes(
                self.__out_types[self.__grid_id]
            ) == LpjmlTypes.LPJ_SHORT:
                read_grid_val = read_short
                type_fact = 0.01
            else:
                read_grid_val = read_float
                type_fact = 1

            for ii in range(0, self.ncell):
                self.grid[1, ii] = read_grid_val(self.channel) * type_fact
                self.grid[2, ii] = read_grid_val(self.channel) * type_fact

    def __write_input_data(self, data, validate_year):
        if not isinstance(data, np.ndarray):
            raise TypeError("Unsupported object type. Please supply a numpy " +
                            "array with the dimension of (ncells, bands).")
        index = read_int(self.channel)
        if not isinstance(data, self.__in_types[index]):
            raise TypeError(
                f"Unsupported data type: {data.dtype} " +
                "Please supply a numpy array with the data type: " +
                f"{self.__in_types[index]}."
            )
        year = read_int(self.channel)
        if not validate_year == year:
            raise ValueError(f"The expected year: {validate_year} does not " +
                             f"match the received year: {year}")
        if index in self.__input_ids.keys():
            bands = getattr(Inputs, self.__input_ids[index]).value
            if not np.shape(data) == (self.ncell, bands):
                ValueError(
                    "The dimensions of the supplied data: " +
                    f"{(self.ncell, bands)} does not match the required " +
                    f"dimensions for {self.__input_ids[index]}: " +
                    f"{(self.ncell, bands)}."
                )
            self.__write_input_values(data)

    def __write_input_values(self, data, cells=None, bands=None, dims=None):
        if not cells and not bands:
            dims = np.shape(data)
            cells = dims[0]
            bands = dims[1]
        write_float(self.channel, data[dims[0]-cells, dims[1]-bands])
        if bands != 0 and cells != 0:
            self.__write_input_values(data=data,
                                      cells=cells,
                                      bands=bands-1,
                                      dims=dims)
        elif bands == 0 and cells != 0:
            self.__write_input_values(data=data,
                                      cells=cells-1,
                                      bands=dims[1],
                                      dims=dims)

    def __read_output_data(self, validate_year):
        index = read_int(self.channel)
        year = read_int(self.channel)
        if not validate_year == year:
            raise ValueError(f"The expected year: {validate_year} does not " +
                             f"match the received year: {year}")
        if index in self.__output_ids.keys():
            bands = self.__out_bands[index]
            out_tmpl = np.zeros(shape=(self.ncell, bands),
                                dtype=self.__out_types[index])
            output = self.__read_output_values(output=out_tmpl,
                                               cells=self.ncell, bands=bands,
                                               dims=np.shape(out_tmpl))
            return output
        else:
            return None

    def __read_output_values(self, output, cells=None, bands=None, dims=None):
        output[dims[0]-cells, dims[0]-bands] = read_float(self.channel)
        if bands != 0 and cells != 0:
            output = self.__read_output_values(output=output,
                                               cells=cells-1,
                                               bands=bands,
                                               dims=dims)
        elif bands != 0 and cells == 0:
            output = self.__read_output_values(output=output,
                                               cells=dims[0],
                                               bands=bands-1,
                                               dims=dims)
        return output
