import os
from datetime import datetime
from subprocess import run, Popen, PIPE, CalledProcessError

import multiprocessing as mp


def operate_lpjml(config_file,
                  model_path,
                  sim_path,
                  std_to_file=False):

    if not os.path.isdir(model_path):
        raise ValueError(
            f"Folder of model_path '{model_path}' does not exist!"
        )

    filename = os.path.splitext(os.path.basename(config_file))[0]
    if filename.startswith("config_"):
        sim_name = filename[7:]
    output_path = f"{sim_path}/output/{sim_name}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stdout_file = os.path.join(output_path, f"stdout_{timestamp}.log")
    stderr_file = os.path.join(output_path, f"stderr_{timestamp}.log")

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        print(f"Created output_path '{output_path}'")

    cmd = [f"{model_path}/bin/lpjml", config_file]
    # environment settings to be used for interartive LPJmL sessions
    #   MPI settings conflict with (e.g. on login node)
    os.environ['I_MPI_DAPL_UD'] = 'disable'
    os.environ['I_MPI_FABRICS'] = 'shm:shm'
    os.environ['I_MPI_DAPL_FABRIC'] = 'shm:sh'
    if std_to_file:
        with open(stdout_file, 'w') as f_out, open(
            stderr_file, 'w'
        ) as f_err:
            with Popen(
                cmd, stdout=f_out, stderr=f_err,
                bufsize=1, universal_newlines=True,
                cwd=model_path
            ) as p:
                p.wait()
    else:
        with Popen(
            cmd, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True,
            cwd=model_path
        ) as p:
            for line in p.stdout:
                print(line, end='')
            for line in p.stderr:
                print(line, end='')

    # reset default MPI settings to be able to submit jobs in parallel again
    os.environ['I_MPI_DAPL_UD'] = 'enable'
    os.environ['I_MPI_FABRICS'] = 'shm:dapl'
    del os.environ['I_MPI_DAPL_FABRIC']
    # raise error if returncode does not reflect successfull call
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)


def run_lpjml(config_file,
              model_path,
              sim_path,
              std_to_file=False):
    """Run LPJmL using a generated (class LpjmlConfig) config file.
    Similar to R function `lpjmlKit::run_lpjml`.
    :param config_file: file name including path if not current to config_file
    :type config_file: str
    :param model_path: path to `LPJmL_internal` (lpjml repository)
    :type model_path: str
    :param sim_path: simulation path to include the output folder where output
        is written to
    :type output_path: str
    :param std_to_file: if True, stdout and stderr are written to files
        in the output folder. Defaults to False.
    :type std_to_file: bool
    """
    run = mp.Process(target=operate_lpjml,
                     args=(config_file, model_path, sim_path, std_to_file))
    run.start()

    return run


def submit_lpjml(config_file,
                 model_path,
                 sim_path=None,
                 group="copan",
                 sclass="short",
                 ntasks=256,
                 wtime=None,
                 dependency=None,
                 blocking=None,
                 couple_to=None):
    """Submit LPJmL run to Slurm using `lpjsubmit` and a generated
    (class LpjmlConfig) config file. Provide arguments for Slurm sbatch
    depending on the run. Similar to R function `lpjmlKit::submit_lpjml`.
    :param config_file: file name including path if not current to config_file
    :type config_file: str
    :param model_path: path to `LPJmL_internal` (lpjml repository)
    :type model_path: str
    :param sim_path: simulation path to include the output folder where output 
        is written to
    :type output_path: str
    :param group: PIK group name to be used for Slurm. Defaults to "copan".
    :type output_path: str
    :param sclass: define the job classification,
        for more information have a look
        [here](https://www.pik-potsdam.de/en/institute/about/it-services/hpc/user-guides/slurm#section-5).
        Defaults to "short".
    :type sclass: str
    :param ntasks: define the number of tasks/threads,
        for more information have a look
        [here](https://www.pik-potsdam.de/en/institute/about/it-services/hpc/user-guides/slurm#section-18).
        Defaults to 256.
    :type ntasks: int/str
    :param wtime: define the time limit which can be an advantage to get faster
        to the top of the (s)queue. For more information have a look
        [here](https://www.pik-potsdam.de/en/institute/about/it-services/hpc/user-guides/slurm#section-18).
    :type wtime: str
    :param dependency: if there is a job that should be processed first (e.g.
        spinup) then pass its job id here.
    :type depdendency: int/str
    :param blocking: cores to be blocked. For more information have a look
        [here](https://www.pik-potsdam.de/en/institute/about/it-services/hpc/user-guides/slurm#section-18).
    :type blocking: int
    :param couple_to: path to program/model/script LPJmL should be coupled to
    :type couple_to: str
    :return: return the submitted jobs id if submitted successfully.
    :rtype: str
    """
    if not os.path.isdir(model_path):
        raise ValueError(
            f"Folder of model_path '{model_path}' does not exist!"
        )
    if not output_path:
        output_path = model_path
    else:
        if not os.path.isdir(output_path):
            raise ValueError(
                f"Folder of output_path '{output_path}' does not exist!"
            )
    # set timestamp with stdout and stderr files to write by batch process
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    stdout = f"{output_path}/outfile_{timestamp}.out"
    stderr = f"{output_path}/errfile_{timestamp}.err"

    lpjroot = os.environ['LPJROOT']
    # prepare lpjsubmit command to be called via subprocess
    cmd = [f"{model_path}/bin/lpjsubmit"]
    # specify sbatch arguments required by lpjsubmit internally
    cmd.extend([
        "-group", group, "-class", sclass, "-o", stdout, "-e", stderr
    ])
    # if dependency (jobid) defined, submit is queued by slurm with nocheck
    if dependency:
        cmd.extend(["-nocheck", "-dependency", str(dependency)])
    # processing time to get a better position in slurm queue
    if wtime:
        cmd.extend(["-wtime", str(wtime)])
    # if cores to be blocked
    if blocking:
        cmd.extend(["-blocking", blocking])
    # run in coupled mode and pass coupling program/model
    if couple_to:
        cmd.extend(["-copan", couple_to])
    cmd.extend([str(ntasks), config_file])
    # set LPJROOT to model_path to be able to call lpjsubmit
    try:
        os.environ['LPJROOT'] = model_path
        # call lpjsubmit via subprocess and return status if successfull
        submit_status = run(cmd, capture_output=True)
    except Exception as e:
        print("Error occurred:", e)
    finally:
        os.environ['LPJROOT'] = lpjroot

    # print stdout and stderr if not successful
    if submit_status.returncode == 0:
        print(submit_status.stdout.decode('utf-8'))
    else:
        print(submit_status.stdout.decode('utf-8'))
        print(submit_status.stderr.decode('utf-8'))
        raise CalledProcessError(submit_status.returncode, submit_status.args)
    # return job id
    return submit_status.stdout.decode('utf-8').split(
        "Submitted batch job "
    )[1].split("\n")[0]


def submit_couple():
    """Start coupled runs of copan:core and LPJmL
    """
    pass
