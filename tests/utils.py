from typing import NotRequired, TypedDict
import struct


class ClmHeader(TypedDict):
    name: NotRequired[str]
    version: NotRequired[int]
    order: NotRequired[int]
    firstyear: NotRequired[int]
    nyear: NotRequired[int]
    firstcell: NotRequired[int]
    ncell: NotRequired[int]
    nbands: NotRequired[int]
    cellsize_lon: NotRequired[float]
    scalar: NotRequired[float]
    cellsize_lat: NotRequired[float]
    datatype: NotRequired[int]
    nstep: NotRequired[int]
    timestep: NotRequired[int]


def clm_file(header: ClmHeader, big_endian: bool = False, data: list[int | float] = []):
    version = header.get('version', 4)
    name = header.get('name', 'LPJGRID')

    struct_string: str = ('>' if big_endian else '<') + f"{len(name)}siiiiiii"

    header_values = [
        name.encode("ascii"),
        version,
        header.get('order', 0 if big_endian else 1),
        header.get('firstyear', 1900),
        header.get('nyear', 1),
        header.get('firstcell', 0),
        header.get('ncell', len(data)),
        header.get('nbands', 1)
    ]

    if version >= 2:
        header_values += [
            header.get('cellsize_lon', 0.5),
            header.get('scalar', 1.0),
        ]
        struct_string += 'ff'
    if version >= 3:
        header_values += [
            header.get('cellsize_lat', 0.5),
            header.get('datatype', 3)  # 3 is float
        ]
        struct_string += 'fi'
    if version >= 4:
        header_values += [
            header.get('nstep', 1),
            header.get('timestep', 1)
        ]
        struct_string += 'ii'
    
    return struct.pack(struct_string, *header_values)


def outputpath_helper(output_dict, path):
    output_dict["file"]["name"] = output_dict["file"]["name"].replace(
        "output/", path + "/"
    )
    return output_dict