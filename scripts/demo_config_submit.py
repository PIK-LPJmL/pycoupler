import os
from pycoupler.utils import check_lpjml, compile_lpjml, clone_lpjml, \
    create_subdirs
from pycoupler.config import read_config
from pycoupler.run import submit_lpjml


# paths
sim_path = "/p/projects/open/Jannes/copan_core/lpjml"
model_path = f"{sim_path}/LPJmL_internal"


# set up lpjml -------------------------------------------------------------- #

# clone function to model location via oauth token (set as enironment var) and
#   checkout copan branch (default until it is merged)
clone_lpjml(model_location=sim_path, branch="master")
# if patched and existing compiled version use make_fast=True or if error is
#   thrown, use arg make_clean=True without make_fast=True
compile_lpjml(model_path=model_path, make_fast=True)
# create required subdirectories to store model related data:
#   restart, output, input
create_subdirs(sim_path)

# define and submit spinup run ---------------------------------------------- #

# create config for spinup run
config_spinup = read_config(file_name=f"{model_path}/lpjml.js", spin_up=True)

# set spinup run configuration
config_spinup.set_spinup(sim_path)

# write config (Config object) as json file
config_spinup_fn = config_spinup.to_json(path=sim_path)

# check if everything is set correct
check_lpjml(config_file=config_spinup_fn, model_path=model_path)
# submit spinup job and get corresponding slurm job id
spinup_jobid = submit_lpjml(
    config_file=config_spinup_fn,
    model_path=model_path,
    sim_path=sim_path
)

# define coupled run -------------------------------------------------------- #

# create config for coupled run
config_coupled = read_config(file_name=f"{model_path}/lpjml.js")

# set coupled run configuration
config_coupled.set_coupled(sim_path,
                           start_year=1901, end_year=2005,
                           coupled_year=1981,
                           coupled_input=["landuse",
                                          "fertilizer_nr"],
                           coupled_output=["cftfrac",
                                           "pft_harvestc",
                                           "pft_harvestn"])

# write config (Config object) as json file
config_coupled_fn = config_coupled.to_json(path=sim_path)

# submit coupled run -------------------------------------------------------- #

# check if everything is set correct
check_lpjml(config_coupled_fn, model_path)
# submit spinup job and get corresponding id
historic_jobid = submit_lpjml(
    config_file=config_coupled_fn,
    model_path=model_path,
    sim_path=sim_path,
    dependency=spinup_jobid,
    couple_to="<COPAN:CORE>"
)
