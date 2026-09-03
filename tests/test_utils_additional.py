"""Additional tests for utils.py functions that need more coverage."""

import json
from unittest.mock import patch

import pytest

from pycoupler.utils import create_subdirs, read_json


class TestCreateSubdirs:
    """Test create_subdirs function."""

    def test_create_subdirs_all_exist(self, tmp_path):
        """Test when all subdirectories already exist."""
        base_path = tmp_path / "base"
        base_path.mkdir()
        (base_path / "input").mkdir()
        (base_path / "output" / "test_sim").mkdir(parents=True)
        (base_path / "restart").mkdir()

        result = create_subdirs(str(base_path), "test_sim")

        assert result == str(base_path)

    def test_create_subdirs_none_exist(self, tmp_path):
        """Test when no subdirectories exist."""
        base_path = tmp_path / "base"
        base_path.mkdir()

        with patch("builtins.print"):
            result = create_subdirs(str(base_path), "test_sim")

        assert result == str(base_path)
        assert (base_path / "input").exists()
        assert (base_path / "output" / "test_sim").exists()
        assert (base_path / "restart").exists()

    def test_create_subdirs_partial_exist(self, tmp_path):
        """Test when some subdirectories exist."""
        base_path = tmp_path / "base"
        base_path.mkdir()
        (base_path / "input").mkdir()
        # output and restart don't exist

        with patch("builtins.print"):
            result = create_subdirs(str(base_path), "test_sim")

        assert result == str(base_path)
        assert (base_path / "input").exists()
        assert (base_path / "output" / "test_sim").exists()
        assert (base_path / "restart").exists()

    def test_create_subdirs_base_path_not_exists(self, tmp_path):
        """Test when base_path doesn't exist."""
        base_path = tmp_path / "nonexistent"

        with pytest.raises(OSError, match="Path.*does not exist"):
            create_subdirs(str(base_path), "test_sim")


class TestReadJson:
    """Test read_json function."""

    def test_read_json_simple(self, tmp_path):
        """Test reading a simple JSON file."""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(data))

        result = read_json(str(json_file))

        assert result == data

    def test_read_json_with_object_hook(self, tmp_path):
        """Test reading JSON with object_hook."""
        json_file = tmp_path / "test.json"
        data = {"key": "value"}
        json_file.write_text(json.dumps(data))

        def custom_hook(dct):
            return {k.upper(): v for k, v in dct.items()}

        result = read_json(str(json_file), object_hook=custom_hook)

        assert result == {"KEY": "value"}

    def test_read_json_nested(self, tmp_path):
        """Test reading nested JSON."""
        json_file = tmp_path / "test.json"
        data = {"level1": {"level2": {"level3": "value"}}, "array": [1, 2, 3]}
        json_file.write_text(json.dumps(data))

        result = read_json(str(json_file))

        assert result == data
        assert result["level1"]["level2"]["level3"] == "value"
        assert result["array"] == [1, 2, 3]

    def test_read_json_file_not_found(self, tmp_path):
        """Test reading non-existent JSON file."""
        json_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            read_json(str(json_file))
