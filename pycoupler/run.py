import os
from datetime import datetime
from subprocess import STDOUT, run, Popen, CalledProcessError
from typing import cast
from pycoupler.config import read_config
import warnings


def operate_lpjml(config_file, std_to_file=False, wait_for_exit=True):
    """Run LPJmL using a generated (class LpjmlConfig) config file.
    Similar to R function `lpjmlKit::run_lpjml`.

    Parameters
    ----------
    config_file : str
        File name including path if not current to config_file
    std_to_file : bool, optional
        If True, stdout and stderr are written to files in the output folder.
        Defaults to False.
    wait_for_exit
        Whether to block the thread until the process exits.
    """

    config = read_config(config_file)

    if config.model_path and not os.path.isdir(config.model_path):
        raise ValueError(f"Folder of model_path '{config.model_path}' does not exist!")

    output_path = f"{config.sim_path}/output/{config.sim_name}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stdout_file = os.path.join(output_path, f"stdout_{timestamp}.log")
    stderr_file = os.path.join(output_path, f"stderr_{timestamp}.log")

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        print(f"Created output_path '{output_path}'")

    subprocess_args = {
        "env": os.environ
        | {
            # environment settings to be used for interactive LPJmL sessions
            #   MPI settings conflict with (e.g. on login node)
            "I_MPI_DAPL_UD": "disable",
            "I_MPI_FABRICS": "shm:shm",
            "I_MPI_DAPL_FABRIC": "shm:sh",
        }
        | config.get_runtime_env(),
        # This might be None, running in the current directory:
        "cwd": config.model_path,
        "text": True,
    }

    if std_to_file:
        subprocess_args |= {
            "stdout": open(stdout_file, "w"),
            "stderr": open(stderr_file, "w"),
            "bufsize": 1,
        }
    else:
        subprocess_args |= {
            "stdout": None,
            "stderr": STDOUT,
            "bufsize": 0,
        }

    p = cast(
        Popen[str],
        config.run_model_bin(
            "lpjml", config_file, detach=True, subprocess_args=subprocess_args
        ),
    )

    if wait_for_exit:
        p.wait()
        if p.stdout:
            p.stdout.close()
        # raise error if returncode does not reflect successfull call
        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)
        return p
    else:
        return p


def run_lpjml(config_file, std_to_file=False):
    """Run LPJmL using a generated (class LpjmlConfig) config file.
    Similar to R function `lpjmlKit::run_lpjml`.

    Parameters
    ----------
    config_file : str
        File name including path if not current to config_file
    std_to_file : bool, optional
        If True, stdout and stderr are written to files in the output folder.
        Defaults to False.
    """
    warnings.warn(
        "run_lpjml is deprecated. Please use operate_lpjml(wait_for_exit=False) instead."
    )
    return operate_lpjml(config_file, std_to_file, False)


def submit_lpjml(
    config_file,
    group="copan",
    sclass="short",
    ntasks=256,
    wtime=None,
    dependency=None,
    blocking=None,
    option=None,
    couple_to=None,
    venv_path=None,
    modules=None,
):
    """Submit LPJmL run to Slurm using `lpjsubmit` and a generated
    (class LpjmlConfig) config file.

    Provide arguments for Slurm sbatch depending on the run.
    Similar to R function `lpjmlKit::submit_lpjml`.

    Parameters
    ----------
    config_file : str
        File name including path if not current to config_file
    group : str, optional
        PIK group name to be used for Slurm. Defaults to "copan".
    sclass : str, optional
        Define the job classification, options are "short", "medium", "long",
        "priority", "standby", "io". For more information have a look at
        <https://www.pik-potsdam.de/en>. Defaults to `"short"`.
    ntasks : int/str, optional
        Define the number of tasks/threads. More information at
        <https://www.pik-potsdam.de/en> and
        <https://slurm.schedmd.com/sbatch.html>. Defaults to 256.
    wtime : str, optional
        Define the time limit. Setting a lower time limit than the maximum
        runtime for `sclass` can reduce the wait time in the SLURM job queue.
        More information at <https://www.pik-potsdam.de/en> and
        <https://slurm.schedmd.com/sbatch.html>.
    dependency : int/str, optional
        If there is a job that should be processed first (e.g. spinup) then pass
        its job id here.
    blocking : int, optional
        Cores to be blocked. More information at
        <https://www.pik-potsdam.de/en> and
        <https://slurm.schedmd.com/sbatch.html>.
    option : str/list, optional
        Additional options to be passed to lpjsubmit. Can be a string or a list
        of strings.
    couple_to : str, optional
        Path to program/model/script LPJmL should be coupled to
    venv_path : str, optional
        Path to a venv to run the coupled script in. This should be the path to
        the top folder of the venv. If not set, `python3` in PATH is used.
    modules : str, optional
        Environment modules to load for the SLURM job separated by spaces.
        For hierarchical modules, observe the necessary module order.

    Returns
    -------
    str
        The submitted jobs id if submitted successfully.
    """

    config = read_config(config_file)
    if not os.path.isdir(config.model_path):
        raise ValueError(f"Folder of model_path '{config.model_path}' does not exist!")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stdout_file = os.path.join(
        config.get_output_folder(ensure=True), f"stdout_{timestamp}.log"
    )
    stderr_file = os.path.join(
        config.get_output_folder(ensure=True), f"stderr_{timestamp}.log"
    )

    # specify sbatch arguments required by lpjsubmit internally
    submit_args = [
        "-group",
        group,
        "-class",
        sclass,
        "-o",
        stdout_file,
        "-e",
        stderr_file,
        # We want to start sbatch ourselves, just generate the job control file
        "-norun",
    ]
    # if dependency (jobid) defined, submit is queued by slurm with nocheck
    if dependency:
        submit_args.extend(["-nocheck", "-dependency", str(dependency)])
    # processing time to get a better position in slurm queue
    if wtime:
        submit_args.extend(["-wtime", str(wtime)])
    # if cores to be blocked
    if blocking:
        submit_args.extend(["-blocking", str(blocking)])

    if option:
        if isinstance(option, str):
            submit_args.extend(["-option", option])
        elif isinstance(option, list):
            for opt in option:
                submit_args.extend(["-option", opt])

    # run in coupled mode and pass coupling program/model
    if couple_to:
        python_path = "python3"
        if venv_path:
            python_path = os.path.join(venv_path, "bin/python")
            if not os.path.isfile(python_path):
                raise FileNotFoundError(
                    f"venv path contains no python binary at '{python_path}'."
                )

        bash_script = f"""#!/bin/bash

# Define the path to the config file
config_file="{config_file}"

# Call the Python script with the config file as an argument
{python_path} {couple_to} $config_file
"""

        couple_file = os.path.join(
            config.get_output_folder(ensure=True), "copan_lpjml.sh"
        )

        with open(couple_file, "w") as file:
            file.write(bash_script)

        # Change the permissions of the file to make it executable
        run(["chmod", "+x", couple_file])

        submit_args.extend(["-couple", couple_file])

    if modules:
        submit_args.extend(["-modules", modules])

    submit_args.extend([str(ntasks), config_file])

    # call lpjsubmit via subprocess and return status if successfull
    submit_file_status = config.run_model_bin(
        "lpjsubmit",
        *submit_args,
        subprocess_args={
            "capture_output": True,
            "cwd": config.sim_path,
            "text": True,
        },
    )

    if submit_file_status.returncode != 0:
        print(submit_file_status.stdout)
        print(submit_file_status.stderr)
        raise CalledProcessError(submit_file_status.returncode, submit_file_status.args)

    sbatch_cmd = ["sbatch"]

    if dependency:
        sbatch_cmd.extend(["-depend", dependency])

    submit_status = run(
        sbatch_cmd,
        cwd=config.sim_path,
        env=submit_env,
        capture_output=True,
        check=True,
        text=True,
    )

    # print stdout and stderr if not successful
    if submit_status.returncode == 0:
        print(submit_status.stdout)
    else:
        print(submit_status.stdout)
        print(submit_status.stderr)
        raise CalledProcessError(submit_status.returncode, submit_status.args)
    # return job id
    return submit_status.stdout.split("Submitted batch job ")[1].split("\n")[0]


def check_lpjml(config_file):
    """Check if config file is set correctly.

    Parameters
    ----------
    config_file : str
        File name (including path) to generated config json file.
    model_path : str
        Path to `LPJmL_internal` (lpjml repository)
    """
    config = read_config(config_file)
    if config.model_path and not os.path.isdir(config.model_path):
        raise ValueError(f"Folder of model_path '{config.model_path}' does not exist!")

    proc_status = config.run_model_bin(
        "lpjcheck",
        [config_file],
        subprocess_args={
            # ensure_paths is false, because this is just a check and should have no side effects
            "cwd": config.model_path,
            "check": False,
            "capture_output": True,
            "text": True,
        },
    )

    if proc_status.returncode == 0:
        print(proc_status.stdout)
    else:
        print(proc_status.stdout)
        print(proc_status.stderr)
