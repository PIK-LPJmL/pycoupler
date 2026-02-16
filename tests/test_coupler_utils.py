"""Test port utility functions from coupler.py."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pycoupler.coupler import (
    kill_process_on_port,
    cleanup_port_on_exit,
    cleanup_port_context,
    safe_port_binding,
)


class TestKillProcessOnPort:
    """Test kill_process_on_port function."""

    @patch("pycoupler.coupler.subprocess.run")
    def test_kill_process_on_port_success(self, mock_run):
        """Test successfully killing processes on a port."""
        # Mock lsof finding two PIDs
        mock_lsof = MagicMock()
        mock_lsof.returncode = 0
        mock_lsof.stdout = "12345\n67890\n"

        # Mock kill commands
        mock_kill = MagicMock()
        mock_kill.returncode = 0

        mock_run.side_effect = [mock_lsof, mock_kill, mock_kill]

        result = kill_process_on_port(8080)

        assert result == 2
        assert mock_run.call_count == 3
        # Check lsof was called correctly
        mock_run.assert_any_call(
            ["lsof", "-ti", ":8080"], capture_output=True, text=True, timeout=5
        )
        # Check kill was called for each PID
        mock_run.assert_any_call(
            ["kill", "-9", "12345"], timeout=5, capture_output=True
        )
        mock_run.assert_any_call(
            ["kill", "-9", "67890"], timeout=5, capture_output=True
        )

    @patch("pycoupler.coupler.subprocess.run")
    def test_kill_process_on_port_no_processes(self, mock_run):
        """Test when no processes are using the port."""
        mock_lsof = MagicMock()
        mock_lsof.returncode = 0
        mock_lsof.stdout = ""  # No PIDs found

        mock_run.return_value = mock_lsof

        result = kill_process_on_port(8080)

        assert result == 0
        assert mock_run.call_count == 1
        mock_run.assert_called_once_with(
            ["lsof", "-ti", ":8080"], capture_output=True, text=True, timeout=5
        )

    @patch("pycoupler.coupler.subprocess.run")
    def test_kill_process_on_port_lsof_fails(self, mock_run):
        """Test when lsof command fails."""
        mock_lsof = MagicMock()
        mock_lsof.returncode = 1  # lsof failed

        mock_run.return_value = mock_lsof

        result = kill_process_on_port(8080)

        assert result == 0
        assert mock_run.call_count == 1

    @patch("pycoupler.coupler.subprocess.run")
    def test_kill_process_on_port_timeout(self, mock_run):
        """Test when lsof times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("lsof", 5)

        result = kill_process_on_port(8080)

        assert result == -1
        assert mock_run.call_count == 1

    @patch("pycoupler.coupler.subprocess.run")
    def test_kill_process_on_port_file_not_found(self, mock_run):
        """Test when lsof command is not found."""
        mock_run.side_effect = FileNotFoundError("lsof not found")

        result = kill_process_on_port(8080)

        assert result == -1
        assert mock_run.call_count == 1

    @patch("pycoupler.coupler.subprocess.run")
    def test_kill_process_on_port_kill_timeout(self, mock_run):
        """Test when kill command times out."""
        mock_lsof = MagicMock()
        mock_lsof.returncode = 0
        mock_lsof.stdout = "12345\n"

        mock_kill_timeout = subprocess.TimeoutExpired("kill", 5)

        mock_run.side_effect = [mock_lsof, mock_kill_timeout]

        result = kill_process_on_port(8080)

        # Should return 0 because no processes were successfully killed
        assert result == 0
        assert mock_run.call_count == 2

    @patch("pycoupler.coupler.subprocess.run")
    def test_kill_process_on_port_whitespace_handling(self, mock_run):
        """Test handling of whitespace in PID output."""
        mock_lsof = MagicMock()
        mock_lsof.returncode = 0
        mock_lsof.stdout = "  12345  \n  67890  \n  "  # Extra whitespace

        mock_kill = MagicMock()
        mock_kill.returncode = 0

        mock_run.side_effect = [mock_lsof, mock_kill, mock_kill]

        result = kill_process_on_port(8080)

        assert result == 2
        # Check that strip() was applied
        mock_run.assert_any_call(
            ["kill", "-9", "12345"], timeout=5, capture_output=True
        )
        mock_run.assert_any_call(
            ["kill", "-9", "67890"], timeout=5, capture_output=True
        )


class TestCleanupPortOnExit:
    """Test cleanup_port_on_exit function."""

    @patch("pycoupler.coupler.atexit.register")
    @patch("pycoupler.coupler.kill_process_on_port")
    def test_cleanup_port_on_exit_registers(self, mock_kill, mock_atexit):
        """Test that cleanup_port_on_exit registers an atexit handler."""
        cleanup_port_on_exit(8080)

        # Verify atexit.register was called
        assert mock_atexit.call_count == 1
        # Get the registered function
        registered_func = mock_atexit.call_args[0][0]
        # Call it to verify it calls kill_process_on_port
        registered_func()
        mock_kill.assert_called_once_with(8080)


class TestCleanupPortContext:
    """Test cleanup_port_context context manager."""

    @patch("pycoupler.coupler.kill_process_on_port")
    def test_cleanup_port_context_success(self, mock_kill):
        """Test successful port cleanup."""
        mock_kill.return_value = 1  # Killed 1 process initially

        with cleanup_port_context(8080) as port:
            assert port == 8080
            # Verify cleanup was called at start
            assert mock_kill.call_count == 1
            mock_kill.assert_called_with(8080)

        # Verify cleanup was called again on exit
        assert mock_kill.call_count == 2

    @patch("pycoupler.coupler.kill_process_on_port")
    def test_cleanup_port_context_exception(self, mock_kill):
        """Test that cleanup happens even when exception occurs."""
        mock_kill.return_value = 0

        try:
            with cleanup_port_context(8080) as port:
                assert port == 8080
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected; cleanup runs in context manager finally block

        # Verify cleanup was called twice (start and finally)
        assert mock_kill.call_count == 2

    @patch("pycoupler.coupler.kill_process_on_port")
    def test_cleanup_port_context_no_existing_processes(self, mock_kill):
        """Test when no processes are using the port."""
        mock_kill.return_value = 0  # No processes killed

        with cleanup_port_context(8080) as port:
            assert port == 8080

        # Cleanup should still be called
        assert mock_kill.call_count == 2

    @patch("pycoupler.coupler.kill_process_on_port")
    def test_cleanup_port_context_multiple_ports(self, mock_kill):
        """Test using multiple ports sequentially."""
        with cleanup_port_context(8080) as port1:
            assert port1 == 8080

        with cleanup_port_context(8081) as port2:
            assert port2 == 8081

        # Each port should have cleanup called twice
        assert mock_kill.call_count == 4


class TestSafePortBinding:
    """Test deprecated safe_port_binding backward compatibility."""

    @patch("pycoupler.coupler.kill_process_on_port")
    def test_safe_port_binding_delegates_to_cleanup_port_context(self, mock_kill):
        """Test that safe_port_binding delegates and ignores host."""
        mock_kill.return_value = 0

        with pytest.warns(DeprecationWarning):
            with safe_port_binding("localhost", 8080) as port:
                assert port == 8080

        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(8080)
