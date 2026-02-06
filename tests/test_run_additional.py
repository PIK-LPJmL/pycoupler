"""Additional tests for run.py functions that need more coverage."""

import os
from unittest.mock import MagicMock, patch, mock_open
from subprocess import CalledProcessError, PIPE

import pytest

from pycoupler.run import operate_lpjml, start_lpjml, run_lpjml
from pycoupler.utils import warn_deprecated_alias


class TestOperateLpjml:
    """Test operate_lpjml function."""

    @patch("pycoupler.run.Popen")
    @patch("pycoupler.run.read_config")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_operate_lpjml_std_to_file(
        self, mock_makedirs, mock_isdir, mock_read_config, mock_popen
    ):
        """Test operate_lpjml with std_to_file=True."""
        # Setup initial environment
        initial_env = {"I_MPI_DAPL_UD": "enable", "I_MPI_FABRICS": "shm:dapl"}
        with patch.dict(os.environ, initial_env, clear=False):
            # Setup mocks
            mock_config = MagicMock()
            mock_config.model_path = "/fake/model/path"
            mock_config.sim_path = "/fake/sim/path"
            mock_config.sim_name = "test_sim"
            mock_read_config.return_value = mock_config
            mock_isdir.return_value = True

            # Mock Popen
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.__enter__ = MagicMock(return_value=mock_process)
            mock_process.__exit__ = MagicMock(return_value=False)
            mock_popen.return_value = mock_process

            # Mock file opening
            with patch("builtins.open", mock_open()):
                operate_lpjml("/fake/config.json", std_to_file=True)

            # Verify Popen was called correctly
            assert mock_popen.called
            call_args = mock_popen.call_args
            assert call_args[1]["cwd"] == "/fake/model/path"
            assert call_args[1]["bufsize"] == 1
            assert call_args[1]["universal_newlines"] is True

            # Verify environment was reset after function completes
            assert os.environ["I_MPI_DAPL_UD"] == "enable"
            assert os.environ["I_MPI_FABRICS"] == "shm:dapl"
            assert "I_MPI_DAPL_FABRIC" not in os.environ

    @patch("pycoupler.run.Popen")
    @patch("pycoupler.run.read_config")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    @patch("os.environ", {"I_MPI_DAPL_UD": "enable", "I_MPI_FABRICS": "shm:dapl"})
    def test_operate_lpjml_std_to_console(
        self, mock_makedirs, mock_isdir, mock_read_config, mock_popen
    ):
        """Test operate_lpjml with std_to_file=False."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_path = "/fake/model/path"
        mock_config.sim_path = "/fake/sim/path"
        mock_config.sim_name = "test_sim"
        mock_read_config.return_value = mock_config
        mock_isdir.return_value = True

        # Mock Popen with stdout/stderr
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ["line1\n", "line2\n"]
        mock_process.stderr = ["error1\n"]
        mock_process.__enter__ = MagicMock(return_value=mock_process)
        mock_process.__exit__ = MagicMock(return_value=False)
        mock_popen.return_value = mock_process

        with patch("builtins.print"):
            operate_lpjml("/fake/config.json", std_to_file=False)

        # Verify Popen was called with PIPE
        call_args = mock_popen.call_args
        assert call_args[1]["stdout"] == PIPE
        assert call_args[1]["stderr"] == PIPE

    @patch("pycoupler.run.Popen")
    @patch("pycoupler.run.read_config")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_operate_lpjml_model_path_not_exists(
        self, mock_makedirs, mock_isdir, mock_read_config, mock_popen
    ):
        """Test operate_lpjml when model_path doesn't exist."""
        mock_config = MagicMock()
        mock_config.model_path = "/fake/model/path"
        mock_read_config.return_value = mock_config
        mock_isdir.return_value = False

        with pytest.raises(ValueError, match="Folder of model_path"):
            operate_lpjml("/fake/config.json")

    @patch("pycoupler.run.Popen")
    @patch("pycoupler.run.read_config")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_operate_lpjml_creates_output_path(
        self, mock_makedirs, mock_isdir, mock_read_config, mock_popen
    ):
        """Test that operate_lpjml creates output path if it doesn't exist."""
        mock_config = MagicMock()
        mock_config.model_path = "/fake/model/path"
        mock_config.sim_path = "/fake/sim/path"
        mock_config.sim_name = "test_sim"
        mock_read_config.return_value = mock_config
        mock_isdir.side_effect = (
            lambda p: p == "/fake/model/path"
        )  # Only model_path exists

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.__enter__ = MagicMock(return_value=mock_process)
        mock_process.__exit__ = MagicMock(return_value=False)
        mock_popen.return_value = mock_process

        with patch("builtins.open", mock_open()), patch("builtins.print"):
            operate_lpjml("/fake/config.json", std_to_file=True)

        # Verify output path was created
        mock_makedirs.assert_called()

    @patch("pycoupler.run.Popen")
    @patch("pycoupler.run.read_config")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_operate_lpjml_process_error(
        self, mock_makedirs, mock_isdir, mock_read_config, mock_popen
    ):
        """Test operate_lpjml when process returns non-zero exit code."""
        mock_config = MagicMock()
        mock_config.model_path = "/fake/model/path"
        mock_config.sim_path = "/fake/sim/path"
        mock_config.sim_name = "test_sim"
        mock_read_config.return_value = mock_config
        mock_isdir.return_value = True

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.args = ["lpjml", "/fake/config.json"]
        mock_process.__enter__ = MagicMock(return_value=mock_process)
        mock_process.__exit__ = MagicMock(return_value=False)
        mock_popen.return_value = mock_process

        with patch("builtins.open", mock_open()), patch("os.environ", {}):
            with pytest.raises(CalledProcessError):
                operate_lpjml("/fake/config.json", std_to_file=True)


class TestStartLpjml:
    """Test start_lpjml function."""

    @patch("multiprocessing.Process")
    def test_start_lpjml(self, mock_process_class):
        """Test start_lpjml creates and starts a process."""
        from pycoupler.run import operate_lpjml

        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        result = start_lpjml("/fake/config.json", std_to_file=True)

        # Verify Process was created with correct target and args
        mock_process_class.assert_called_once_with(
            target=operate_lpjml, args=("/fake/config.json", True)
        )
        # Verify process was started
        mock_process.start.assert_called_once()
        # Verify correct process was returned
        assert result == mock_process


class TestRunLpjml:
    """Test run_lpjml deprecated alias."""

    @patch("pycoupler.run.start_lpjml")
    @patch("pycoupler.run.warn_deprecated_alias")
    def test_run_lpjml_calls_start_lpjml(self, mock_warn, mock_start):
        """Test that run_lpjml calls start_lpjml and emits deprecation warning."""
        from pycoupler.run import start_lpjml

        mock_start.return_value = MagicMock()

        result = run_lpjml("/fake/config.json", std_to_file=False)

        # Verify deprecation warning was emitted
        mock_warn.assert_called_once_with(start_lpjml, "run_lpjml", "start_lpjml")
        # Verify start_lpjml was called with correct args
        mock_start.assert_called_once_with("/fake/config.json", std_to_file=False)
        # Verify result is returned
        assert result == mock_start.return_value


class TestWarnDeprecatedAlias:
    """Test warn_deprecated_alias function."""

    def test_warn_deprecated_alias_callable(self):
        """Test warning for callable (function)."""

        def test_func():
            pass

        with pytest.warns(DeprecationWarning, match="run_lpjml is deprecated"):
            warn_deprecated_alias(test_func, "run_lpjml", "start_lpjml")

    def test_warn_deprecated_alias_instance(self):
        """Test warning for instance (class method)."""

        class TestClass:
            def method(self):
                pass

        instance = TestClass()

        with pytest.warns(
            DeprecationWarning, match="TestClass.old_method is deprecated"
        ):
            warn_deprecated_alias(instance, "old_method", "new_method")
