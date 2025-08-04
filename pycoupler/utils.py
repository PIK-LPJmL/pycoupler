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
    return {
        "Afghanistan": {"name": "Afghanistan", "code": "AFG"},
        "Aland Islands": {"name": "Aland Islands", "code": "ALA"},
        "Albania": {"name": "Albania", "code": "ALB"},
        "Algeria": {"name": "Algeria", "code": "DZA"},
        "American Samoa": {"name": "American Samoa", "code": "ASM"},
        "Angola": {"name": "Angola", "code": "AGO"},
        "Anguilla": {"name": "Anguilla", "code": "AIA"},
        "Antigua and Barbuda": {
            "name": "Antigua and Barbuda",
            "code": "ATG",
        },
        "Argentina": {"name": "Argentina", "code": "ARG"},
        "Armenia": {"name": "Armenia", "code": "ARM"},
        "Austria": {"name": "Austria", "code": "AUT"},
        "Azerbaijan": {"name": "Azerbaijan", "code": "AZE"},
        "Bahamas The": {"name": "Bahamas, The", "code": "BHS"},
        "Bahrain": {"name": "Bahrain", "code": "BHR"},
        "Bangladesh": {"name": "Bangladesh", "code": "BGD"},
        "Barbados": {"name": "Barbados", "code": "BRB"},
        "Belgium": {"name": "Belgium", "code": "BEL"},
        "Belize": {"name": "Belize", "code": "BLZ"},
        "Benin": {"name": "Benin", "code": "BEN"},
        "Bermuda": {"name": "Bermuda", "code": "BMU"},
        "Bhutan": {"name": "Bhutan", "code": "BTN"},
        "Bolivia": {"name": "Bolivia", "code": "BOL"},
        "Bosnia and Herzegovina": {
            "name": "Bosnia and Herzegovina",
            "code": "BIH",
        },
        "Botswana": {"name": "Botswana", "code": "BWA"},
        "British Indian Ocean Territory": {
            "name": "British Indian Ocean Territory",
            "code": "IOT",
        },
        "Brunei": {"name": "Brunei", "code": "BRN"},
        "Bulgaria": {"name": "Bulgaria", "code": "BGR"},
        "Burkina Faso": {"name": "Burkina Faso", "code": "BFA"},
        "Burundi": {"name": "Burundi", "code": "BDI"},
        "Byelarus": {"name": "Byelarus", "code": "BLR"},
        "Cambodia": {"name": "Cambodia", "code": "KHM"},
        "Cameroon": {"name": "Cameroon", "code": "CMR"},
        "Cape Verde": {"name": "Cape Verde", "code": "CPV"},
        "Cayman Islands": {"name": "Cayman Islands", "code": "CYM"},
        "Central African Republic": {
            "name": "Central African Republic",
            "code": "CAF",
        },
        "Chad": {"name": "Chad", "code": "TCD"},
        "Chile": {"name": "Chile", "code": "CHL"},
        "Christmas Island": {"name": "Christmas Island", "code": "CXR"},
        "Cocos Keeling Islands": {
            "name": "Cocos Keeling Islands",
            "code": "CCK",
        },
        "Colombia": {"name": "Colombia", "code": "COL"},
        "Comoros": {"name": "Comoros", "code": "COM"},
        "Congo Brazzaville": {"name": "Congo-Brazzaville", "code": "COG"},
        "Cook Islands": {"name": "Cook Islands", "code": "COK"},
        "Costa Rica": {"name": "Costa Rica", "code": "CRI"},
        "Croatia": {"name": "Croatia", "code": "HRV"},
        "Cuba": {"name": "Cuba", "code": "CUB"},
        "Curacao": {"name": "Curacao", "code": "CUW"},
        "Cyprus": {"name": "Cyprus", "code": "CYP"},
        "Czech Republic": {"name": "Czech Republic", "code": "CZE"},
        "Denmark": {"name": "Denmark", "code": "DNK"},
        "Djibouti": {"name": "Djibouti", "code": "DJI"},
        "Dominica": {"name": "Dominica", "code": "DMA"},
        "Dominican Republic": {"name": "Dominican Republic", "code": "DOM"},
        "Ecuador": {"name": "Ecuador", "code": "ECU"},
        "Egypt": {"name": "Egypt", "code": "EGY"},
        "El Salvador": {"name": "El Salvador", "code": "SLV"},
        "Equatorial Guinea": {"name": "Equatorial Guinea", "code": "GNQ"},
        "Eritrea": {"name": "Eritrea", "code": "ERI"},
        "Estonia": {"name": "Estonia", "code": "EST"},
        "Ethiopia": {"name": "Ethiopia", "code": "ETH"},
        "Falkland Islands or Islas Malvinas": {
            "name": "Falkland Islands or Islas Malvinas",
            "code": "FLK",
        },
        "Faroe Islands": {"name": "Faroe Islands", "code": "FRO"},
        "Federated States of Micronesia": {
            "name": "Federated States of Micronesia",
            "code": "FSM",
        },
        "Fiji": {"name": "Fiji", "code": "FJI"},
        "Finland": {"name": "Finland", "code": "FIN"},
        "France": {"name": "France", "code": "FRA"},
        "French Guiana": {"name": "French Guiana", "code": "GUF"},
        "French Polynesia": {"name": "French Polynesia", "code": "PYF"},
        "French Southern and Antarctica Lands": {
            "name": "French Southern and Antarctica Lands",
            "code": "NOC",
        },
        "Gabon": {"name": "Gabon", "code": "GAB"},
        "Gambia The": {"name": "Gambia,The", "code": "GMB"},
        "Georgia": {"name": "Georgia", "code": "GEO"},
        "Germany": {"name": "Germany", "code": "DEU"},
        "Ghana": {"name": "Ghana", "code": "GHA"},
        "Greece": {"name": "Greece", "code": "GRC"},
        "Greenland": {"name": "Greenland", "code": "GRL"},
        "Grenada": {"name": "Grenada", "code": "GRD"},
        "Guadeloupe": {"name": "Guadeloupe", "code": "GLP"},
        "Guam": {"name": "Guam", "code": "GUM"},
        "Guatemala": {"name": "Guatemala", "code": "GTM"},
        "Guernsey": {"name": "Guernsey", "code": "GGY"},
        "Guinea Bissau": {"name": "Guinea-Bissau", "code": "GNB"},
        "Guinea": {"name": "Guinea", "code": "GIN"},
        "Guyana": {"name": "Guyana", "code": "GUY"},
        "Haiti": {"name": "Haiti", "code": "HTI"},
        "Heard Island and McDonald Islands": {
            "name": "Heard Island and McDonald Islands",
            "code": "HMD",
        },
        "Honduras": {"name": "Honduras", "code": "HND"},
        "Hong Kong": {"name": "Hong Kong", "code": "HKG"},
        "Hungary": {"name": "Hungary", "code": "HUN"},
        "Iceland": {"name": "Iceland", "code": "ISL"},
        "Indonesia": {"name": "Indonesia", "code": "IDN"},
        "Iran": {"name": "Iran", "code": "IRN"},
        "Iraq": {"name": "Iraq", "code": "IRQ"},
        "Ireland": {"name": "Ireland", "code": "IRL"},
        "Isle of Man": {"name": "Isle of Man", "code": "IMN"},
        "Israel": {"name": "Israel", "code": "ISR"},
        "Italy": {"name": "Italy", "code": "ITA"},
        "Ivory Coast": {"name": "Ivory Coast", "code": "CIV"},
        "Jamaica": {"name": "Jamaica", "code": "JAM"},
        "Japan": {"name": "Japan", "code": "JPN"},
        "Jersey": {"name": "Jersey", "code": "JEY"},
        "Jordan": {"name": "Jordan", "code": "JOR"},
        "Kazakhstan": {"name": "Kazakhstan", "code": "KAZ"},
        "Kenya": {"name": "Kenya", "code": "KEN"},
        "Kiribati": {"name": "Kiribati", "code": "KIR"},
        "Kosovo": {"name": "Kosovo", "code": "KO-"},
        "Kuwait": {"name": "Kuwait", "code": "KWT"},
        "Kyrgyzstan": {"name": "Kyrgyzstan", "code": "KGZ"},
        "Laos": {"name": "Laos", "code": "LAO"},
        "Latvia": {"name": "Latvia", "code": "LVA"},
        "Lebanon": {"name": "Lebanon", "code": "LBN"},
        "Lesotho": {"name": "Lesotho", "code": "LSO"},
        "Liberia": {"name": "Liberia", "code": "LBR"},
        "Libya": {"name": "Libya", "code": "LBY"},
        "Lithuania": {"name": "Lithuania", "code": "LTU"},
        "Luxembourg": {"name": "Luxembourg", "code": "LUX"},
        "Macedonia": {"name": "Macedonia", "code": "MKD"},
        "Madagascar": {"name": "Madagascar", "code": "MDG"},
        "Malawi": {"name": "Malawi", "code": "MWI"},
        "Malaysia": {"name": "Malaysia", "code": "MYS"},
        "Maldives": {"name": "Maldives", "code": "MDV"},
        "Mali": {"name": "Mali", "code": "MLI"},
        "Malta": {"name": "Malta", "code": "MLT"},
        "Marshall Islands": {"name": "Marshall Islands", "code": "MHL"},
        "Martinique": {"name": "Martinique", "code": "MTQ"},
        "Mauritania": {"name": "Mauritania", "code": "MRT"},
        "Mauritius": {"name": "Mauritius", "code": "MUS"},
        "Mayotte": {"name": "Mayotte", "code": "MYT"},
        "Mexico": {"name": "Mexico", "code": "MEX"},
        "Moldova": {"name": "Moldova", "code": "MDA"},
        "Mongolia": {"name": "Mongolia", "code": "MNG"},
        "Montenegro": {"name": "Montenegro", "code": "MNE"},
        "Montserrat": {"name": "Montserrat", "code": "MSR"},
        "Morocco": {"name": "Morocco", "code": "MAR"},
        "Mozambique": {"name": "Mozambique", "code": "MOZ"},
        "Myanmar or Burma": {"name": "Myanmar or Burma", "code": "MMR"},
        "Namibia": {"name": "Namibia", "code": "NAM"},
        "Nauru": {"name": "Nauru", "code": "NRU"},
        "Nepal": {"name": "Nepal", "code": "NPL"},
        "Netherlands": {"name": "Netherlands", "code": "NLD"},
        "New Caledonia": {"name": "New Caledonia", "code": "NCL"},
        "New Zealand": {"name": "New Zealand", "code": "NZL"},
        "Nicaragua": {"name": "Nicaragua", "code": "NIC"},
        "Niger": {"name": "Niger", "code": "NER"},
        "Nigeria": {"name": "Nigeria", "code": "NGA"},
        "Niue": {"name": "Niue", "code": "NIU"},
        "No Land": {"name": "No Land", "code": "XNL"},
        "Norfolk Island": {"name": "Norfolk Island", "code": "NFK"},
        "North Korea": {"name": "North Korea", "code": "PRK"},
        "Northern Mariana Islands": {
            "name": "Northern Mariana Islands",
            "code": "MNP",
        },
        "Norway": {"name": "Norway", "code": "NOR"},
        "Oman": {"name": "Oman", "code": "OMN"},
        "Pakistan": {"name": "Pakistan", "code": "PAK"},
        "Palau": {"name": "Palau", "code": "PLW"},
        "Panama": {"name": "Panama", "code": "PAN"},
        "Papua New Guinea": {"name": "Papua New Guinea", "code": "PNG"},
        "Paraguay": {"name": "Paraguay", "code": "PRY"},
        "Peru": {"name": "Peru", "code": "PER"},
        "Philippines": {"name": "Philippines", "code": "PHL"},
        "Pitcairn Islands": {"name": "Pitcairn Islands", "code": "PCN"},
        "Poland": {"name": "Poland", "code": "POL"},
        "Portugal": {"name": "Portugal", "code": "PRT"},
        "Puerto Rico": {"name": "Puerto Rico", "code": "PRI"},
        "Qatar": {"name": "Qatar", "code": "QAT"},
        "Reunion": {"name": "Reunion", "code": "REU"},
        "Romania": {"name": "Romania", "code": "ROU"},
        "Rwanda": {"name": "Rwanda", "code": "RWA"},
        "Saint Helena Ascension and Tristan da Cunha": {
            "name": "Saint Helena Ascension and Tristan da Cunha",
            "code": "SHN",
        },
        "Saint Kitts and Nevis": {
            "name": "Saint Kitts and Nevis",
            "code": "KNA",
        },
        "Saint Lucia": {"name": "Saint Lucia", "code": "LCA"},
        "Saint Pierre and Miquelon": {
            "name": "Saint Pierre and Miquelon",
            "code": "SPM",
        },
        "Sao Tome and Principe": {
            "name": "Sao Tome and Principe",
            "code": "STP",
        },
        "Saudi Arabia": {"name": "Saudi Arabia", "code": "SAU"},
        "Senegal": {"name": "Senegal", "code": "SEN"},
        "Serbia": {"name": "Serbia", "code": "SRB"},
        "Seychelles": {"name": "Seychelles", "code": "SYC"},
        "Sierra Leone": {"name": "Sierra Leone", "code": "SLE"},
        "Singapore": {"name": "Singapore", "code": "SGP"},
        "Slovakia": {"name": "Slovakia", "code": "SVK"},
        "Slovenia": {"name": "Slovenia", "code": "SVN"},
        "Solomon Islands": {"name": "Solomon Islands", "code": "SLB"},
        "Somalia": {"name": "Somalia", "code": "SOM"},
        "South Africa": {"name": "South Africa", "code": "ZAF"},
        "South Georgia and the South Sandwich Islands": {
            "name": "South Georgia and the South Sandwich Islands",
            "code": "SGS",
        },
        "South Korea": {"name": "South Korea", "code": "KOR"},
        "South Sudan": {"name": "South Sudan", "code": "SSD"},
        "Spain": {"name": "Spain", "code": "ESP"},
        "Sri Lanka": {"name": "Sri Lanka", "code": "LKA"},
        "St Vincent and the Grenadines": {
            "name": "St. Vincent and the Grenadines",
            "code": "VCT",
        },
        "Sudan": {"name": "Sudan", "code": "SDN"},
        "Suriname": {"name": "Suriname", "code": "SUR"},
        "Svalbard": {"name": "Svalbard", "code": "SJM"},
        "Swaziland": {"name": "Swaziland", "code": "SWZ"},
        "Sweden": {"name": "Sweden", "code": "SWE"},
        "Switzerland": {"name": "Switzerland", "code": "CHE"},
        "Syria": {"name": "Syria", "code": "SYR"},
        "Taiwan": {"name": "Taiwan", "code": "TWN"},
        "Tajikistan": {"name": "Tajikistan", "code": "TJK"},
        "Tanzania United Republic of": {
            "name": "Tanzania, United Republic of",
            "code": "TZA",
        },
        "Thailand": {"name": "Thailand", "code": "THA"},
        "Timor Leste": {"name": "Timor Leste", "code": "TLS"},
        "Togo": {"name": "Togo", "code": "TGO"},
        "Tokelau": {"name": "Tokelau", "code": "TKL"},
        "Tonga": {"name": "Tonga", "code": "TON"},
        "Trinidad and Tobago": {"name": "Trinidad and Tobago", "code": "TTO"},
        "Tunisia": {"name": "Tunisia", "code": "TUN"},
        "Turkey": {"name": "Turkey", "code": "TUR"},
        "Turkmenistan": {"name": "Turkmenistan", "code": "TKM"},
        "Turks and Caicos Islands": {
            "name": "Turks and Caicos Islands",
            "code": "TCA",
        },
        "Tuvalu": {"name": "Tuvalu", "code": "TUV"},
        "Uganda": {"name": "Uganda", "code": "UGA"},
        "Ukraine": {"name": "Ukraine", "code": "UKR"},
        "United Arab Emirates": {"name": "United Arab Emirates", "code": "ARE"},
        "United Kingdom": {"name": "United Kingdom", "code": "GBR"},
        "United States Minor Outlying Islands": {
            "name": "United States Minor Outlying Islands",
            "code": "UMI",
        },
        "Uruguay": {"name": "Uruguay", "code": "URY"},
        "Uzbekistan": {"name": "Uzbekistan", "code": "UZB"},
        "Vanuatu": {"name": "Vanuatu", "code": "VUT"},
        "Venezuela": {"name": "Venezuela", "code": "VEN"},
        "Vietnam": {"name": "Vietnam", "code": "VNM"},
        "Virgin Islands": {"name": "Virgin Islands", "code": "VGB"},
        "Wallis and Futuna": {"name": "Wallis and Futuna", "code": "WLF"},
        "West Bank": {"name": "West Bank", "code": "PSE"},
        "Western Sahara": {"name": "Western Sahara", "code": "ESH"},
        "Western Samoa": {"name": "Western Samoa", "code": "WSM"},
        "Yemen": {"name": "Yemen", "code": "YEM"},
        "Zaire DR Congo": {"name": "DR Congo, former Zaire", "code": "COD"},
        "Zambia": {"name": "Zambia", "code": "ZMB"},
        "Zimbabwe": {"name": "Zimbabwe", "code": "ZWE"},
        "Australia": {"name": "Australia", "code": "AUS"},
        "Brazil": {"name": "Brazil", "code": "BRA"},
        "Canada": {"name": "Canada", "code": "CAN"},
        "China": {"name": "China", "code": "CHN"},
        "India": {"name": "India", "code": "IND"},
        "Russia": {"name": "Russia", "code": "RUS"},
        "United States": {"name": "United States of America", "code": "USA"},
    }


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
