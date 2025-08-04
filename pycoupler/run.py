import os
from datetime import datetime
from subprocess import run, Popen, PIPE, CalledProcessError
from pycoupler.config import read_config

import multiprocessing as mp


def operate_lpjml(config_file, std_to_file=False):
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

    config = read_config(config_file)

    if not os.path.isdir(config.model_path):
        raise ValueError(f"Folder of model_path '{config.model_path}' does not exist!")

    output_path = f"{config.sim_path}/output/{config.sim_name}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stdout_file = os.path.join(output_path, f"stdout_{timestamp}.log")
    stderr_file = os.path.join(output_path, f"stderr_{timestamp}.log")

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        print(f"Created output_path '{output_path}'")

    cmd = [f"{config.model_path}/bin/lpjml", config_file]
    # environment settings to be used for interartive LPJmL sessions
    #   MPI settings conflict with (e.g. on login node)
    os.environ["I_MPI_DAPL_UD"] = "disable"
    os.environ["I_MPI_FABRICS"] = "shm:shm"
    os.environ["I_MPI_DAPL_FABRIC"] = "shm:sh"
    if std_to_file:
        with open(stdout_file, "w") as f_out, open(stderr_file, "w") as f_err:
            with Popen(
                cmd,
                stdout=f_out,
                stderr=f_err,
                bufsize=1,
                universal_newlines=True,
                cwd=config.model_path,
            ) as p:
                p.wait()
    else:
        with Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            bufsize=1,
            universal_newlines=True,
            cwd=config.model_path,
        ) as p:
            for line in p.stdout:
                print(line, end="")
            for line in p.stderr:
                print(line, end="")

    # reset default MPI settings to be able to submit jobs in parallel again
    os.environ["I_MPI_DAPL_UD"] = "enable"
    os.environ["I_MPI_FABRICS"] = "shm:dapl"
    del os.environ["I_MPI_DAPL_FABRIC"]
    # raise error if returncode does not reflect successfull call
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)


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
    run = mp.Process(target=operate_lpjml, args=(config_file, std_to_file))
    run.start()

    return run


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

    Returns
    -------
    str
        The submitted jobs id if submitted successfully.
    """

    config = read_config(config_file)
    if not os.path.isdir(config.model_path):
        raise ValueError(f"Folder of model_path '{config.model_path}' does not exist!")

    output_path = f"{config.sim_path}/output/{config.sim_name}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stdout_file = os.path.join(output_path, f"stdout_{timestamp}.log")
    stderr_file = os.path.join(output_path, f"stderr_{timestamp}.log")

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        print(f"Created output_path '{output_path}'")

    lpjroot = os.environ.get("LPJROOT")
    # prepare lpjsubmit command to be called via subprocess
    cmd = [f"{config.model_path}/bin/lpjsubmit"]
    # specify sbatch arguments required by lpjsubmit internally
    cmd.extend(
        [
            "-group",
            group,
            "-class",
            sclass,
            "-o",
            stdout_file,
            "-e",
            stderr_file,
        ]  # noqa: E501
    )
    # if dependency (jobid) defined, submit is queued by slurm with nocheck
    if dependency:
        cmd.extend(["-nocheck", "-dependency", str(dependency)])
    # processing time to get a better position in slurm queue
    if wtime:
        cmd.extend(["-wtime", str(wtime)])
    # if cores to be blocked
    if blocking:
        cmd.extend(["-blocking", str(blocking)])

    if option:
        if isinstance(option, str):
            cmd.extend(["-option", option])
        elif isinstance(option, list):
            for opt in option:
                cmd.extend(["-option", opt])

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

        couple_file = f"{output_path}/copan_lpjml.sh"

        with open(couple_file, "w") as file:
            file.write(bash_script)

        # Change the permissions of the file to make it executable
        run(["chmod", "+x", couple_file])

        cmd.extend(["-couple", couple_file])

    cmd.extend([str(ntasks), config_file])

    # Intialize submit_status in higher scope
    submit_status = None
    # set LPJROOT to model_path to be able to call lpjsubmit
    try:
        os.environ["LPJROOT"] = config.model_path
        # call lpjsubmit via subprocess and return status if successfull
        submit_status = run(cmd, capture_output=True)
    except Exception as e:
        print("Error occurred:", e)
    finally:
        if lpjroot:
            os.environ["LPJROOT"] = lpjroot
        else:
            del os.environ["LPJROOT"]

    # print stdout and stderr if not successful
    if submit_status is None:
        raise Exception("Process was not submitted.")
    elif submit_status.returncode == 0:
        print(submit_status.stdout.decode("utf-8"))
    else:
        print(submit_status.stdout.decode("utf-8"))
        print(submit_status.stderr.decode("utf-8"))
        raise CalledProcessError(submit_status.returncode, submit_status.args)
    # return job id
    return (
        submit_status.stdout.decode("utf-8")
        .split("Submitted batch job ")[1]
        .split("\n")[0]
    )


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
    if not os.path.isdir(config.model_path):
        raise ValueError(f"Folder of model_path '{config.model_path}' does not exist!")
    if os.path.isfile(f"{config.model_path}/bin/lpjcheck"):
        proc_status = run(
            ["./bin/lpjcheck", config_file],
            capture_output=True,  # "-param",
            cwd=config.model_path,
        )
    if proc_status.returncode == 0:
        print(proc_status.stdout.decode("utf-8"))
    else:
        print(proc_status.stdout.decode("utf-8"))
        print(proc_status.stderr.decode("utf-8"))
