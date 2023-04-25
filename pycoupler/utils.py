import os
from subprocess import run, Popen, PIPE, CalledProcessError


def clone_lpjml(model_location=".", branch="lpjml53_copan"):
    """Git clone lpjml via oauth using git url and token provided as
    environment variables. If copan implementation still on branch use branch
    argument.
    :param model_location: location to `git clone` lpjml -> LPJmL_internaö
    :type model_location: str
    :param branch: switch/`git checkout` to branch with copan implementation.
        Defaults to "lpjml53_copan".
    :type branch: str
    """
    git_url = os.environ.get("GIT_LPJML_URL")
    git_token = os.environ.get("GIT_READ_TOKEN")
    cmd = ["git", "clone", f"https://oauth2:{git_token}@{git_url}"]
    with Popen(
        cmd, stdout=PIPE, bufsize=1, universal_newlines=True,
        cwd=model_location
    ) as p:
        for line in p.stdout:
            print(line, end='')
    # raise error if returncode does not reflect successfull call
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)
    # check if branch required
    if branch:
        with Popen(
            ["git", "checkout", branch],
            stdout=PIPE, bufsize=1, universal_newlines=True,
            cwd=f"{model_location}/LPJmL_internal"
        ) as p:
            for line in p.stdout:
                print(line, end='')


def compile_lpjml(model_path=".", make_fast=False, make_clean=False):
    """Compile or make lpjml after model clone/changes. make_fast for small
    changes, make_clean to delete previous compiled version (clean way)
    :param model_path: path to `LPJmL_internal` (lpjml repository)
    :type model_path: str
    :param make_fast: make with arg -j8. Defaults to False.
    :type make_fast: bool
    :param make_clean: delete previous compiled model version. Defaults to
        False.
    :type make_clean: bool
    """
    if not os.path.isdir(model_path):
        raise ValueError(
            f"Folder of model_path '{model_path}' does not exist!"
        )
    if not os.path.isfile(f"{model_path}/bin/lpjml"):
        proc_status = run(
            "./configure.sh", capture_output=True, cwd=model_path
        )
        print(proc_status.stdout.decode('utf-8'))
    # make clean first
    if make_clean:
        run(["make", "clean"], capture_output=True, cwd=model_path)
    # make all call with possibility to make fast via -j8 arg
    cmd = ['make']
    if make_fast:
        cmd.append('-j16')
    cmd.append('all')
    # open process to be iteratively printed to the console
    with Popen(
        cmd, stdout=PIPE, bufsize=1, universal_newlines=True, cwd=model_path
    ) as p:
        for line in p.stdout:
            print(line, end='')
    # raise error if returncode does not reflect successfull call
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)


def check_lpjml(config_file, model_path):
    """Check if config file is set correctly.
    :param config_file: file_name (including path) to generated config json
        file.
    :type model_path: str
    :param model_path: path to `LPJmL_internal` (lpjml repository)
    :type model_path: str
    """
    if not os.path.isdir(model_path):
        raise ValueError(
            f"Folder of model_path '{model_path}' does not exist!"
        )
    if os.path.isfile(f"{model_path}/bin/lpjcheck"):
        proc_status = run(
            ["./bin/lpjcheck", config_file], capture_output=True,  # "-param",
            cwd=model_path
        )
    if proc_status.returncode == 0:
        print(proc_status.stdout.decode('utf-8'))
    else:
        print(proc_status.stdout.decode('utf-8'))
        print(proc_status.stderr.decode('utf-8'))


def create_subdirs(base_path):
    """Check if config file is set correctly.
    :param base_path: directory to check wether required subfolders exists. If
        not create corresponding folder (input, output, restart)
    :type base_path: str
    """
    if not os.path.exists(base_path):
        raise OSError(f"Path '{base_path}' does not exist.")

    if not os.path.exists(f"{base_path}/input"):
        os.makedirs(f"{base_path}/input")
        print(f"Input path '{base_path}/input' was created.")

    if not os.path.exists(f"{base_path}/output"):
        os.makedirs(f"{base_path}/output")
        print(f"Output path '{base_path}/output' was created.")

    if not os.path.exists(f"{base_path}/restart"):
        os.makedirs(f"{base_path}/restart")
        print(f"Restart path '{base_path}/restart' was created.")
