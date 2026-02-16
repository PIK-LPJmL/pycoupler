"""Test the utils functions."""

import pytest
from pycoupler.utils import search_country, detect_io_type


def test_search_country():

    germany = search_country("germany")
    assert germany == "DEU"

    france = search_country("franz")
    assert france == "FRA"

    netherlands = search_country("nether")
    assert netherlands == "NLD"


def test_detect_io_type(test_path, tmp_path):
    # Test meta file detection
    meta_file = detect_io_type(f"{test_path}/data/output/coupled_test/grid.nc4.json")
    assert meta_file == "meta"

    # Test netcdf file detection
    netcdf_file = detect_io_type(f"{test_path}/data/output/coupled_test/grid.nc4")
    assert netcdf_file == "cdf"

    # Test binary file with header (clm) detection
    clm_file = detect_io_type(f"{test_path}/data/input/coord_netherlands.clm")
    assert clm_file == "clm"

    # Test binary file without header (raw) detection
    raw_file = detect_io_type(f"{test_path}/data/input/soil_netherlands.bin")
    assert raw_file == "raw"

    # Test text file detection
    csv_file = detect_io_type(f"{test_path}/data/output/coupled_test/inseeds_data.csv")
    assert csv_file == "text"

    # Test invalid file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        detect_io_type(f"{test_path}/data/non_existent_file.txt")

    # Test binary file that can't be decoded as UTF-8 (triggers UnicodeDecodeError)
    # Create a binary file that's not valid UTF-8 (use tmp_path to avoid writing
    # into the checked-in tests/data directory)
    binary_file = tmp_path / "invalid_utf8.bin"
    binary_file.write_bytes(b"\xff\xfe\x00\x01")  # Invalid UTF-8 sequence
    result = detect_io_type(str(binary_file))
    assert result == "raw"  # Should default to 'raw' when decode fails
