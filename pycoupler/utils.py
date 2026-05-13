import importlib.resources
import os
import json
from fuzzywuzzy import fuzz, process


def get_countries():
    """Current workaround to get countries defined in LPJmL.

    Returns
    -------
    dict
        Dictionary with countries and their codes.
    """
    with (importlib.resources.files(__package__) / "countries.json").open(
        "r"
    ) as countries:
        return json.load(countries)


def search_country(query):
    """Search for countries based on a fuzzy matching algorithm.

    Parameters
    ----------
    query : str
        The search query.

    Returns
    -------
    str
        The matching country code.
    """
    countries = get_countries()
    name, _ = process.extractOne(query, countries.keys(), scorer=fuzz.ratio)
    return countries[name]["code"]


def read_json(file_name, object_hook=None):
    with open(file_name) as file_con:
        json_dict = json.load(file_con, object_hook=object_hook)

    return json_dict


def create_subdirs(base_path, sim_name):
    """Check if config file is set correctly.

    Parameters
    ----------
    base_path : str
        Directory to check wether required subfolders exists. If not create
        corresponding folder (input, output, restart)
    sim_name : str
        Name of the simulation. Used to create output folder.

    Returns
    -------
    str
        base_path
    """
    if not os.path.exists(base_path):
        raise OSError(f"Path '{base_path}' does not exist.")

    if not os.path.exists(f"{base_path}/input"):
        os.makedirs(f"{base_path}/input")
        print(f"Input path '{base_path}/input' was created.")

    if not os.path.exists(f"{base_path}/output/{sim_name}"):
        os.makedirs(f"{base_path}/output/{sim_name}")
        print(f"Output path '{base_path}/output/{sim_name}' was created.")

    if not os.path.exists(f"{base_path}/restart"):
        os.makedirs(f"{base_path}/restart")
        print(f"Restart path '{base_path}/restart' was created.")

    return base_path


def detect_io_type(filename):
    """
    Detect the file type of an LPJmL input/output file.

    Parameters
    ----------
    filename : str
        Path to the file to check.

    Returns
    -------
    str
        Detected file type ('cdf', 'clm', 'meta', 'raw', or 'text').

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")

    # Read the first 10 bytes of the file
    with open(filename, "rb") as f:
        file_check = f.read(min(os.path.getsize(filename), 10))

    # Check for 'clm' (LPJmL binary format with header)
    if len(file_check) >= 3 and file_check[:3] == b"LPJ":
        return "clm"

    # Check for NetCDF format
    if (len(file_check) >= 3 and file_check[:3] == b"CDF") or (
        len(file_check) >= 8 and file_check[:8] == b"\x89HDF\r\n\x1a\n"
    ):
        return "cdf"

    # Check if file is a text file
    try:
        text_content = file_check.decode("utf-8")
        if all(32 <= ord(c) <= 126 or c in "\r\n\t" for c in text_content):
            # Check if it is a JSON file (starts with '{' after stripping
            # whitespace)
            if text_content.lstrip().startswith("{"):
                return "meta"
            return "text"
    except UnicodeDecodeError:
        pass  # Not a valid UTF-8 text file

    # Default to 'raw' if no other type is detected
    return "raw"
