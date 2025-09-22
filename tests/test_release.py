#!/usr/bin/env python3
"""
Unit tests for the pycoupler.release module.

Tests the unified release functionality including:
- Package name detection
- Git branch detection
- CITATION.cff file updates
- Command execution and error handling
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

from pycoupler.release import (
    get_package_name,
    get_current_branch,
    update_citation_file,
    run_command,
    main,
)

# Add the pycoupler module to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGetPackageName:
    """Test package name detection functionality."""

    def test_get_package_name_from_pyproject_toml(self):
        """Test package name detection from pyproject.toml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock pyproject.toml
            pyproject_content = """
[project]
name = "test-package"
version = "1.0.0"
"""
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text(pyproject_content)

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = get_package_name()
                assert result == "test-package"
            finally:
                os.chdir(original_cwd)

    def test_get_package_name_fallback_to_directory(self):
        """Test package name fallback to directory name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory with a specific name
            test_dir = Path(temp_dir) / "my-awesome-package"
            test_dir.mkdir()

            original_cwd = os.getcwd()
            try:
                os.chdir(test_dir)
                result = get_package_name()
                assert result == "my-awesome-package"
            finally:
                os.chdir(original_cwd)

    def test_get_package_name_with_invalid_pyproject(self):
        """Test package name detection with invalid pyproject.toml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an invalid pyproject.toml
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text("invalid toml content")

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = get_package_name()
                assert result == Path(temp_dir).name
            finally:
                os.chdir(original_cwd)


class TestGetCurrentBranch:
    """Test Git branch detection functionality."""

    @patch("subprocess.run")
    def test_get_current_branch_success(self, mock_run):
        """Test successful branch detection."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        result = get_current_branch()
        assert result == "main"
        mock_run.assert_called_once_with(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_current_branch_failure(self, mock_run):
        """Test branch detection failure fallback."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")

        result = get_current_branch()
        assert result == "main"  # fallback value

    @patch("subprocess.run")
    def test_get_current_branch_file_not_found(self, mock_run):
        """Test branch detection when git is not found."""
        mock_run.side_effect = FileNotFoundError()

        result = get_current_branch()
        assert result == "main"  # fallback value


class TestUpdateCitationFile:
    """Test CITATION.cff file update functionality."""

    def test_update_citation_file_version_only(self):
        """Test updating only the version in CITATION.cff."""
        with tempfile.TemporaryDirectory() as temp_dir:
            citation_file = Path(temp_dir) / "CITATION.cff"
            citation_content = """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
- family-names: "Test"
  given-names: "Author"
title: "Test Package"
version: 1.0.0
date-released: 2023-01-01
"""
            citation_file.write_text(citation_content)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = update_citation_file("2.0.0")
                assert result is True

                # Check the file was updated
                updated_content = citation_file.read_text()
                assert "version: 2.0.0" in updated_content
                assert "version: 1.0.0" not in updated_content
            finally:
                os.chdir(original_cwd)

    def test_update_citation_file_with_date(self):
        """Test updating both version and date in CITATION.cff."""
        with tempfile.TemporaryDirectory() as temp_dir:
            citation_file = Path(temp_dir) / "CITATION.cff"
            citation_content = """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
- family-names: "Test"
  given-names: "Author"
title: "Test Package"
version: 1.0.0
date-released: 2023-01-01
"""
            citation_file.write_text(citation_content)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = update_citation_file("2.0.0", "2024-01-01")
                assert result is True

                # Check the file was updated
                updated_content = citation_file.read_text()
                assert "version: 2.0.0" in updated_content
                assert "date-released: 2024-01-01" in updated_content
            finally:
                os.chdir(original_cwd)

    def test_update_citation_file_no_changes_needed(self):
        """Test when no changes are needed to CITATION.cff."""
        with tempfile.TemporaryDirectory() as temp_dir:
            citation_file = Path(temp_dir) / "CITATION.cff"
            citation_content = """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
- family-names: "Test"
  given-names: "Author"
title: "Test Package"
version: 2.0.0
date-released: 2024-01-01
"""
            citation_file.write_text(citation_content)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = update_citation_file("2.0.0", "2024-01-01")
                assert result is False  # No changes needed
            finally:
                os.chdir(original_cwd)

    def test_update_citation_file_missing_file(self):
        """Test when CITATION.cff file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = update_citation_file("2.0.0")
                assert result is False  # No file to update
            finally:
                os.chdir(original_cwd)

    def test_update_citation_file_encoding_error(self):
        """Test handling of encoding errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            citation_file = Path(temp_dir) / "CITATION.cff"
            # Create a file with invalid encoding
            citation_file.write_bytes(b"invalid \xff content")

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = update_citation_file("2.0.0")
                assert result is False  # Should handle error gracefully
            finally:
                os.chdir(original_cwd)


class TestRunCommand:
    """Test command execution functionality."""

    @patch("subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value = MagicMock(returncode=0)

        result = run_command("echo test", "Testing command")
        assert result is True
        mock_run.assert_called_once_with("echo test", shell=True, check=True)

    @patch("subprocess.run")
    def test_run_command_failure_with_exit(self, mock_run):
        """Test command failure with exit."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "test")

        with pytest.raises(SystemExit) as exc_info:
            run_command("echo test", "Testing command", fail_on_error=True)
        assert exc_info.value.code == 1

    @patch("subprocess.run")
    def test_run_command_failure_without_exit(self, mock_run):
        """Test command failure without exit."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "test")

        result = run_command("echo test", "Testing command", fail_on_error=False)
        assert result is False


class TestMain:
    """Test main function functionality."""

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.update_citation_file")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    @patch("subprocess.run")
    def test_main_success(
        self,
        mock_subprocess,
        mock_branch,
        mock_package,
        mock_update_citation,
        mock_run_command,
    ):
        """Test successful main function execution."""
        # Setup mocks
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"
        mock_update_citation.return_value = True
        mock_run_command.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock sys.argv
        with patch.object(sys, "argv", ["release.py", "1.0.0"]):
            with patch("builtins.print") as mock_print:
                main()

        # Verify calls
        assert mock_run_command.call_count == 4  # black, pytest, flake8, git add
        mock_update_citation.assert_called_once_with("1.0.0")

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    def test_main_help(self, mock_branch, mock_package, mock_run_command):
        """Test main function help display."""
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"

        # Test --help
        with patch.object(sys, "argv", ["release.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        # Test -h
        with patch.object(sys, "argv", ["release.py", "-h"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    def test_main_no_version(self, mock_branch, mock_package, mock_run_command):
        """Test main function with no version provided."""
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"

        with patch.object(sys, "argv", ["release.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0  # Help should be shown

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    def test_main_black_failure(self, mock_branch, mock_package, mock_run_command):
        """Test main function when black formatting fails."""
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"

        # Make black fail
        def side_effect(cmd, desc, fail_on_error=True):
            if "black" in cmd:
                if fail_on_error:
                    raise SystemExit(1)
                return False
            return True

        mock_run_command.side_effect = side_effect

        with patch.object(sys, "argv", ["release.py", "1.0.0"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestIntegration:
    """Integration tests for the release module."""

    def test_full_release_workflow_mock(self):
        """Test the full release workflow with mocked dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock CITATION.cff
            citation_file = Path(temp_dir) / "CITATION.cff"
            citation_content = """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
- family-names: "Test"
  given-names: "Author"
title: "Test Package"
version: 1.0.0
date-released: 2023-01-01
"""
            citation_file.write_text(citation_content)

            # Create a mock pyproject.toml
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_content = """
[project]
name = "test-package"
version = "1.0.0"
"""
            pyproject_file.write_text(pyproject_content)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Mock all external dependencies
                with patch("pycoupler.release.run_command") as mock_run_command, patch(
                    "subprocess.run"
                ) as mock_subprocess:

                    mock_run_command.return_value = True
                    mock_subprocess.return_value = MagicMock(returncode=0)

                    # Test the main function
                    with patch.object(sys, "argv", ["release.py", "2.0.0"]):
                        with patch("builtins.print"):
                            main()

                    # Verify that all expected commands were called
                    assert mock_run_command.call_count >= 3  # black, pytest, flake8

                    # Verify CITATION.cff was updated
                    updated_content = citation_file.read_text()
                    assert "version: 2.0.0" in updated_content

            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__])
