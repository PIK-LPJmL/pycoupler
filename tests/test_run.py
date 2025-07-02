from pycoupler.run import submit_lpjml
import pytest


class TestLpjSubmit:
    group = "copan"
    sclass = "short"
    ntasks = 256
    wtime = "00:16:10"
    couple_script = "/some/path/to/script.py"

    @pytest.fixture(autouse=True)
    def mock_lpjsubmit(self, fp):
        # Allow other commands, such as chmod
        fp.allow_unregistered(True)
        # Register a fake process for lpjsubmit (see https://pytest-subprocess.readthedocs.io/en/latest/usage.html#non-exact-command-matching)
        fp.register(
            [fp.program("lpjsubmit"), fp.any()],
            stdout="Mock lpjsubmit\nSubmitted batch job 42\nsome stuff",
        )

    @pytest.fixture(autouse=True, params=[True, False], ids=["with_venv", "no_venv"])
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
            venv_path=str(mock_venv) if request.param else None,
        )

    def test_job_id(self, submit):
        assert submit == "42"

    def test_command(self, sim_path, config_coupled, fp):
        run_script_path = sim_path / "output/coupled_test/inseeds.sh"
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

    def test_run_script(self, sim_path, config_coupled, mock_venv, request):

        run_script_path = sim_path / "output/coupled_test/inseeds.sh"
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
{f"{str(mock_venv)}/bin/python" if "with_venv" in request.node.name else "python3"} {self.couple_script} $config_file
"""
            )
