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
    @patch("pycoupler.release.get_current_version")
    @patch("subprocess.run")
    def test_main_success(
        self,
        mock_subprocess,
        mock_current_version,
        mock_branch,
        mock_package,
        mock_update_citation,
        mock_run_command,
    ):
        """Test successful main function execution."""
        # Setup mocks
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"
        mock_current_version.return_value = "0.9.0"  # Valid previous version
        mock_update_citation.return_value = True
        mock_run_command.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock sys.argv
        with patch.object(sys, "argv", ["release.py", "1.0.0"]):
            with patch("builtins.print") as mock_print:
                main()

        # Verify calls
        # black, pytest, flake8, git add, git push (remote tag deletion)
        assert mock_run_command.call_count == 5
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


class TestVersionValidation:
    """Test version validation functionality."""

    def test_parse_version_valid(self):
        """Test parsing valid version strings."""
        from pycoupler.release import parse_version

        assert parse_version("1.0.0") == (1, 0, 0)
        assert parse_version("2.5.10") == (2, 5, 10)
        assert parse_version("0.1.0") == (0, 1, 0)

    def test_parse_version_invalid(self):
        """Test parsing invalid version strings."""
        from pycoupler.release import parse_version

        assert parse_version("1.0") is None
        assert parse_version("1.0.0.0") is None
        assert parse_version("v1.0.0") is None
        assert parse_version("1.0.0-beta") is None
        assert parse_version("invalid") is None

    def test_is_valid_version_increment_no_current(self):
        """Test version validation when no current version exists."""
        from pycoupler.release import is_valid_version_increment

        # Any valid semantic version should be allowed
        assert is_valid_version_increment(None, "1.0.0") is True
        assert is_valid_version_increment(None, "0.1.0") is True
        assert is_valid_version_increment(None, "2.5.10") is True

        # Invalid versions should be rejected
        assert is_valid_version_increment(None, "1.0") is False
        assert is_valid_version_increment(None, "v1.0.0") is False

    def test_is_valid_version_increment_same_version(self):
        """Test that same version is allowed (re-release)."""
        from pycoupler.release import is_valid_version_increment

        assert is_valid_version_increment("1.0.0", "1.0.0") is True
        assert is_valid_version_increment("2.5.10", "2.5.10") is True

    def test_is_valid_version_increment_patch(self):
        """Test valid patch version increments."""
        from pycoupler.release import is_valid_version_increment

        # Valid patch increments
        assert is_valid_version_increment("1.0.0", "1.0.1") is True
        assert is_valid_version_increment("2.5.10", "2.5.11") is True

        # Invalid patch increments
        assert is_valid_version_increment("1.0.0", "1.0.2") is False  # Skip
        assert is_valid_version_increment("1.0.0", "1.0.0") is True  # Same (re-release)

    def test_is_valid_version_increment_minor(self):
        """Test valid minor version increments."""
        from pycoupler.release import is_valid_version_increment

        # Valid minor increments
        assert is_valid_version_increment("1.0.0", "1.1.0") is True
        assert is_valid_version_increment("2.5.10", "2.6.0") is True

        # Invalid minor increments
        assert is_valid_version_increment("1.0.0", "1.2.0") is False  # Skip
        assert is_valid_version_increment("1.0.0", "1.1.1") is False  # Should be patch

    def test_is_valid_version_increment_major(self):
        """Test valid major version increments."""
        from pycoupler.release import is_valid_version_increment

        # Valid major increments
        assert is_valid_version_increment("1.0.0", "2.0.0") is True
        assert is_valid_version_increment("2.5.10", "3.0.0") is True

        # Invalid major increments
        assert is_valid_version_increment("1.0.0", "3.0.0") is False  # Skip
        assert is_valid_version_increment("1.0.0", "2.1.0") is False  # Should be minor

    def test_is_valid_version_increment_backwards(self):
        """Test that backwards version increments are invalid."""
        from pycoupler.release import is_valid_version_increment

        # Backwards increments should be invalid
        assert is_valid_version_increment("2.0.0", "1.0.0") is False
        assert is_valid_version_increment("1.1.0", "1.0.0") is False
        assert is_valid_version_increment("1.0.1", "1.0.0") is False

    def test_get_current_version(self):
        """Test getting current version from Git tags."""
        from pycoupler.release import get_current_version
        from unittest.mock import patch

        # Test with valid tag
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="v1.5.0\n", returncode=0)
            assert get_current_version() == "1.5.0"

        # Test with tag without 'v' prefix
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="1.5.0\n", returncode=0)
            assert get_current_version() == "1.5.0"

        # Test with no tags
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            assert get_current_version() is None


class TestReReleaseConfirmation:
    """Test re-release confirmation functionality."""

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.update_citation_file")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    @patch("pycoupler.release.get_current_version")
    @patch("subprocess.run")
    def test_release_confirmation_yes(
        self,
        mock_subprocess,
        mock_current_version,
        mock_branch,
        mock_package,
        mock_update_citation,
        mock_run_command,
    ):
        """Test re-release confirmation with 'yes' response."""
        # Setup mocks
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"
        mock_current_version.return_value = "1.0.0"  # Same as requested version
        mock_update_citation.return_value = True
        mock_run_command.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock sys.argv and input
        with patch.object(sys, "argv", ["release.py", "1.0.0"]):
            with patch("builtins.print"):
                with patch("builtins.input", return_value="y"):
                    main()

        # Verify that the release process continued
        assert mock_run_command.call_count >= 3  # black, pytest, flake8

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.update_citation_file")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    @patch("pycoupler.release.get_current_version")
    @patch("subprocess.run")
    def test_release_confirmation_no(
        self,
        mock_subprocess,
        mock_current_version,
        mock_branch,
        mock_package,
        mock_update_citation,
        mock_run_command,
    ):
        """Test re-release confirmation with 'no' response."""
        # Setup mocks
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"
        mock_current_version.return_value = "1.0.0"  # Same as requested version
        mock_update_citation.return_value = True
        mock_run_command.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock sys.argv and input
        with patch.object(sys, "argv", ["release.py", "1.0.0"]):
            with patch("builtins.print"):
                with patch("builtins.input", return_value="n"):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0  # Should exit with 0 (cancelled)

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.update_citation_file")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    @patch("pycoupler.release.get_current_version")
    @patch("subprocess.run")
    def test_release_confirmation_invalid_then_valid(
        self,
        mock_subprocess,
        mock_current_version,
        mock_branch,
        mock_package,
        mock_update_citation,
        mock_run_command,
    ):
        """Test re-release confirmation with invalid then valid response."""
        # Setup mocks
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"
        mock_current_version.return_value = "1.0.0"  # Same as requested version
        mock_update_citation.return_value = True
        mock_run_command.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock sys.argv and input (invalid then valid)
        with patch.object(sys, "argv", ["release.py", "1.0.0"]):
            with patch("builtins.print"):
                with patch("builtins.input", side_effect=["maybe", "y"]):
                    main()

        # Verify that the release process continued
        assert mock_run_command.call_count >= 3  # black, pytest, flake8

    @patch("pycoupler.release.run_command")
    @patch("pycoupler.release.update_citation_file")
    @patch("pycoupler.release.get_package_name")
    @patch("pycoupler.release.get_current_branch")
    @patch("pycoupler.release.get_current_version")
    @patch("subprocess.run")
    def test_no_confirmation_for_new_version(
        self,
        mock_subprocess,
        mock_current_version,
        mock_branch,
        mock_package,
        mock_update_citation,
        mock_run_command,
    ):
        """Test that no confirmation is needed for new versions."""
        # Setup mocks
        mock_package.return_value = "test-package"
        mock_branch.return_value = "main"
        mock_current_version.return_value = "1.0.0"  # Different from requested version
        mock_update_citation.return_value = True
        mock_run_command.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock sys.argv (no input needed)
        with patch.object(sys, "argv", ["release.py", "1.0.1"]):
            with patch("builtins.print"):
                main()

        # Verify that the release process continued without asking for confirmation
        assert mock_run_command.call_count >= 3  # black, pytest, flake8


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
                ) as mock_subprocess, patch(
                    "pycoupler.release.get_current_version"
                ) as mock_current_version:

                    mock_run_command.return_value = True
                    mock_subprocess.return_value = MagicMock(returncode=0)
                    mock_current_version.return_value = (
                        "1.0.0"  # Valid previous version
                    )

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
