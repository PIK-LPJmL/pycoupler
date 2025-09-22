from pycoupler.run import submit_lpjml
import pytest
from subprocess import CalledProcessError
import pytest_subprocess  # noqa: F401


class TestLpjSubmit:
    group = "copan"
    sclass = "short"
    ntasks = 256
    wtime = "00:16:10"
    couple_script = "/some/path/to/script.py"

    @pytest.fixture(autouse=True)
    def mock_lpjsubmit(self, fp, request):
        # We expect chmod to actually modify permissions
        fp.pass_command([fp.program("chmod"), "+x", fp.any(min=1, max=1)])
        if hasattr(request, "param") and request.param == "no mocking":
            return
        # Register a fake process for lpjsubmit
        # (see https://pytest-subprocess.readthedocs.io/en/latest/usage.html#non-exact-command-matching) # noqa: E501
        return fp.register(
            [fp.program("lpjsubmit"), fp.any()],
            stdout="Mock lpjsubmit\nSubmitted batch job 42\nsome stuff",
            returncode=(
                1
                if hasattr(request, "param") and request.param == "non-zero errorcode"
                else 0
            ),
        )

    @pytest.fixture()
    def mock_venv(self, tmp_path_factory, request):
        if hasattr(request, "param") and request.param == "none":
            return None
        else:
            venv = tmp_path_factory.mktemp("venv")
            if not hasattr(request, "param") or request.param != "broken":
                (venv / "bin").mkdir()
                (venv / "bin" / "python").touch()
            return str(venv)

    @pytest.fixture(autouse=True)
    def submit(
        self,
        mock_venv,
        sim_path,
        config_coupled,
        request,
    ):
        return submit_lpjml(
            config_coupled,
            group=self.group,
            sclass=self.sclass,
            ntasks=self.ntasks,
            wtime=self.wtime,
            couple_to=self.couple_script,
            venv_path=mock_venv,
        )

    def test_job_id(self, submit):
        assert submit == "42"

    @pytest.mark.parametrize(
        "mock_lpjsubmit",
        [
            pytest.param(
                "no mocking",
                marks=pytest.mark.xfail(raises=Exception),
            ),
            pytest.param(
                "non-zero errorcode",
                marks=pytest.mark.xfail(raises=CalledProcessError),
            ),
        ],
        indirect=True,
    )
    def test_lpjsubmit_error_cases(self, mock_lpjsubmit):
        # The test does nothing, we expect the fail in the fixtures
        pass

    def test_command(self, sim_path, config_coupled, fp, submit):
        run_script_path = sim_path / "output/coupled_test/copan_lpjml.sh"
        assert (
            fp.call_count(
                [
                    fp.program("lpjsubmit"),
                    "-group",
                    self.group,
                    "-class",
                    self.sclass,
                    "-o",
                    fp.any(max=1, min=1),
                    "-e",
                    fp.any(max=1, min=1),
                    "-wtime",
                    self.wtime,
                    "-couple",
                    str(run_script_path),
                    str(self.ntasks),
                    config_coupled,
                ]
            )
            == 1
        ), "lpjsubmit should be called exactly once with correct parameters"

    @pytest.mark.parametrize(
        "mock_venv",
        [
            "working",
            pytest.param("broken", marks=pytest.mark.xfail(raises=FileNotFoundError)),
            "none",
        ],
        indirect=True,
    )
    def test_run_script(self, sim_path, config_coupled, mock_venv, request, submit):
        run_script_path = sim_path / "output/coupled_test/copan_lpjml.sh"
        assert run_script_path.is_file(), "run script should have been created"
        assert (
            run_script_path.stat().st_mode & 0o0100
        ), "run script should be executable"
        with run_script_path.open("r") as f:
            assert (
                f.read()
                == f"""#!/bin/bash

# Define the path to the config file
config_file="{config_coupled}"

# Call the Python script with the config file as an argument
{f"{mock_venv}/bin/python" if mock_venv else "python3"} {self.couple_script} \
$config_file
"""
            )
