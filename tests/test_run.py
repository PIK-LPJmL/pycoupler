import sys
from pycoupler.run import submit_lpjml
import pytest
from subprocess import CalledProcessError
import pytest_subprocess  # noqa: F401


class TestLpjSubmit:
    group = "copan"
    sclass = "short"
    ntasks = 256
    wtime = "00:16:10"

    @pytest.fixture
    def couple_script(self, sim_path):
        script_path = sim_path / "model.py"
        script_path.touch()
        return script_path

    @pytest.fixture(autouse=True)
    def mock_lpjsubmit(self, fp, sim_path, request):
        if getattr(request, "param", None) == "no mocking":
            return
        # Register a fake process for lpjsubmit
        # (see https://pytest-subprocess.readthedocs.io/en/latest/usage.html#non-exact-command-matching) # noqa: E501

        def fake_lpjsubmit(process, exit_code):
            process.returncode = 1 if exit_code == "non-zero errorcode" else 0
            (sim_path / "slurm.jcf").touch()

        return fp.register(
            [fp.program("lpjsubmit"), fp.any()],
            stdout="Mock lpjsubmit\nSubmitted batch job 42\nsome stuff",
            callback=fake_lpjsubmit,
            callback_kwargs={"exit_code": getattr(request, "param", None)},
        )

    @pytest.fixture(autouse=True)
    def mock_sbatch(self, fp, request):
        # We expect chmod to actually modify permissions
        if hasattr(request, "param") and request.param == "no mocking":
            return
        # Register a fake process for lpjsubmit
        # (see https://pytest-subprocess.readthedocs.io/en/latest/usage.html#non-exact-command-matching) # noqa: E501
        return fp.register(
            [fp.program("sbatch"), fp.any()],
            stdout="Submitted batch job 42",
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
        couple_script,
        config_coupled_json,
        request,
    ):
        return submit_lpjml(
            config_coupled_json,
            group=self.group,
            sclass=self.sclass,
            ntasks=self.ntasks,
            wtime=self.wtime,
            couple_to=couple_script,
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

    @pytest.mark.parametrize(
        "mock_venv",
        [
            "working",
            pytest.param("broken", marks=pytest.mark.xfail(raises=FileNotFoundError)),
            "none",
        ],
        indirect=True,
    )
    def test_command(
        self, sim_path, config_coupled_json, fp, mock_venv, couple_script, submit
    ):
        assert (
            fp.call_count(
                [
                    fp.program("lpjsubmit"),
                    "-o",
                    fp.any(max=1, min=1),
                    "-e",
                    fp.any(max=1, min=1),
                    "-norun",
                    "-group",
                    self.group,
                    "-class",
                    self.sclass,
                    "-wtime",
                    self.wtime,
                    "-couple",
                    f"{f"{mock_venv}/bin/python" if mock_venv else sys.executable} {couple_script} {config_coupled_json}",
                    str(self.ntasks),
                    config_coupled_json,
                ]
            )
            == 1
        ), "lpjsubmit should be called exactly once with correct parameters"
        assert (
            fp.call_count(
                [fp.program("sbatch"), str(sim_path / "slurm_coupled_test.jcf")]
            )
            == 1
        ), "sbatch should be called exactly once with correct parameters"
