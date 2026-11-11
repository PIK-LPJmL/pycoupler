from typing import TypedDict  # , NotRequired TODO: Was added in python 3.11
import struct


class ClmHeader(TypedDict):
    name: str  # NotRequired[str]
    version: int  # NotRequired[int]
    order: int  # NotRequired[int]
    firstyear: int  # NotRequired[int]
    nyear: int  # NotRequired[int]
    firstcell: int  # NotRequired[int]
    ncell: int  # NotRequired[int]
    nbands: int  # NotRequired[int]
    cellsize_lon: float  # NotRequired[float]
    scalar: float  # NotRequired[float]
    cellsize_lat: float  # NotRequired[float]
    datatype: int  # NotRequired[int]
    nstep: int  # NotRequired[int]
    timestep: int  # NotRequired[int]


def clm_file(
    header: ClmHeader, big_endian: bool = False, data=[]
):  # data: list[int | float]
    version = header.get("version", 4)
    name = header.get("name", "LPJGRID")

    struct_string: str = (">" if big_endian else "<") + f"{len(name)}siiiiiii"

    header_values = [
        name.encode("ascii"),
        version,
        header.get("order", 0 if big_endian else 1),
        header.get("firstyear", 1900),
        header.get("nyear", 1),
        header.get("firstcell", 0),
        header.get("ncell", len(data)),
        header.get("nbands", 1),
    ]

    if version >= 2:
        header_values += [
            header.get("cellsize_lon", 0.5),
            header.get("scalar", 1.0),
        ]
        struct_string += "ff"
    if version >= 3:
        header_values += [
            header.get("cellsize_lat", 0.5),
            header.get("datatype", 3),  # 3 is float
        ]
        struct_string += "fi"
    if version >= 4:
        header_values += [header.get("nstep", 1), header.get("timestep", 1)]
        struct_string += "ii"

    return struct.pack(struct_string, *header_values)


def outputpath_helper(output_dict, path):
    output_dict["file"]["name"] = output_dict["file"]["name"].replace(
        "output/", path + "/"
    )
    return output_dict
