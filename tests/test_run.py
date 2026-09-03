import json
from pathlib import Path
from subprocess import CalledProcessError
import pytest_subprocess  # noqa: F401

import pytest

from pycoupler.run import submit_lpjml


class TestLpjSubmit:
    group = "copan"
    sclass = "short"
    ntasks = 256
    wtime = "00:16:10"
    couple_script = "/some/path/to/script.py"
    sbatch_job_id = "4242"

    @pytest.fixture()
    def slurm_wait_state(self, request):
        return getattr(request, "param", "missing")

    @pytest.fixture(autouse=True)
    def mock_lpjsubmit(self, fp, request, slurm_wait_state, config_coupled, tmp_path):
        # We expect chmod to actually modify permissions
        fp.pass_command([fp.program("chmod"), "+x", fp.any(min=1, max=1)])
        fail_mode = getattr(request, "param", None)
        if fail_mode == "no mocking":
            return

        slurm_jcf_path = tmp_path / "slurm.jcf"

        def _lpjsubmit_callback(_):
            slurm_text = self._build_slurm_text(
                config_coupled,
                has_wait=(slurm_wait_state == "present"),
            )
            slurm_jcf_path.write_text(slurm_text)

        fp.register(
            [fp.program("lpjsubmit"), fp.any()],
            stdout="Mock lpjsubmit\nSubmitted batch job 41\nsome stuff",
            returncode=(1 if fail_mode == "non-zero errorcode" else 0),
            callback=_lpjsubmit_callback,
        )

        fp.register(
            ["sbatch", str(slurm_jcf_path)],
            stdout=f"Submitted batch job {self.sbatch_job_id}\n",
        )
        return slurm_jcf_path

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
        tmp_path,
    ):
        return submit_lpjml(
            config_coupled,
            group=self.group,
            sclass=self.sclass,
            ntasks=self.ntasks,
            wtime=self.wtime,
            couple_to=self.couple_script,
            venv_path=mock_venv,
            slurm_jcf_dir=tmp_path,
        )

    def test_job_id(self, submit):
        assert submit == self.sbatch_job_id

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
                    "-norun",
                    str(self.ntasks),
                    config_coupled,
                ]
            )
            == 1
        ), "lpjsubmit should be called exactly once with correct parameters"

    def test_sbatch_called(self, fp, submit, tmp_path):
        slurm_jcf_path = tmp_path / "slurm.jcf"
        assert (
            fp.call_count(["sbatch", str(slurm_jcf_path)]) == 1
        ), "sbatch should be invoked once with generated slurm.jcf"

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
            assert f.read() == f"""#!/bin/bash

# Define the path to the config file
config_file="{config_coupled}"

# Call the Python script with the config file as an argument
{f"{mock_venv}/bin/python" if mock_venv else "python3"} {self.couple_script} \
$config_file
"""

    def test_slurm_wait_block_injected(self, config_coupled, submit, tmp_path):
        slurm_text = (tmp_path / "slurm.jcf").read_text()
        assert "couple_pid=$!" in slurm_text
        assert "wait $couple_pid" in slurm_text

    @pytest.mark.parametrize("slurm_wait_state", ["present"], indirect=True)
    def test_slurm_wait_block_respected(
        self, config_coupled, submit, slurm_wait_state, tmp_path
    ):
        expected = self._build_slurm_text(config_coupled, has_wait=True)
        assert (tmp_path / "slurm.jcf").read_text() == expected

    def _build_slurm_text(self, config_path: str, has_wait: bool) -> str:
        couple_file = self._couple_file(config_path)
        base = (
            "#!/bin/bash\n\n"
            f"{couple_file}  &\n\n"
            "mpirun $LPJROOT/bin/lpjml args\n\n"
            "rc=$?\n"
            "exit $rc # exit with return code\n"
        )
        if has_wait:
            base = base.replace(
                f"{couple_file}  &\n\n",
                f"{couple_file}  &\ncouple_pid=$!\n\n",
                1,
            )
            base = base.replace(
                "exit $rc # exit with return code",
                'if [ -n "${couple_pid:-}" ]; then\n'
                "  wait $couple_pid\n"
                "  couple_rc=$?\n"
                "  if [ $rc -eq 0 ]; then\n"
                "    rc=$couple_rc\n"
                "  fi\n"
                "fi\n"
                "exit $rc # exit with return code",
                1,
            )
        return base

    def _couple_file(self, config_path: str) -> str:
        with open(config_path) as fh:
            cfg = json.load(fh)
        return str(
            Path(cfg["sim_path"]) / "output" / cfg["sim_name"] / "copan_lpjml.sh"
        )
