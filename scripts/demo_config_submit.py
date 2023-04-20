import os
from pycoupler.utils import check_lpjml, compile_lpjml, clone_lpjml, \
    create_subdirs
from pycoupler.config import parse_config
from pycoupler.run import run_lpjml


# paths
base_path = "<INSERT_MODEL_LOCATION>"
model_path = f"{base_path}/LPJmL_internal"

output_path = f"{base_path}/output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

restart_path = f"{base_path}/restart"
if not os.path.exists(restart_path):
    os.makedirs(restart_path)

# set up lpjml -------------------------------------------------------------- #

# clone function to model location via oauth token (set as enironment var) and
#   checkout copan branch (default until it is merged)
clone_lpjml(model_location=base_path, branch="master")
# if patched and existing compiled version use make_fast=True or if error is
#   thrown, use arg make_clean=True without make_fast=True
compile_lpjml(model_path=model_path, make_fast=True)
# create required subdirectories to store model related data:
#   restart, output, input
create_subdirs(base_path)

# define and submit spinup run ---------------------------------------------- #

# create config for spinup run
config_spinup = parse_config(path=model_path, spin_up=True)
# set spinup run configuration
config_spinup.set_spinup(output_path, restart_path)
# write config (LpjmlConfig object) as json file
config_spinup_fn = f"{base_path}/config_spinup.json"
config_spinup.to_json(file=config_spinup_fn)

# check if everything is set correct
check_lpjml(config_file=config_spinup_fn, model_path=model_path)
# submit spinup job and get corresponding slurm job id
spinup_jobid = submit_lpjml(
    config_file=config_spinup_fn, model_path=model_path,
    output_path=output_path
)


# define and submit historic run -------------------------------------------- #

# create config for historic run
config_historic = parse_config(path=model_path)
# set historic run configuration
config_historic.set_historic(output_path, restart_path, start=1901, end=1980,
                             write_start=1980)  # write_start=1980
# write config (LpjmlConfig object) as json file
config_historic_fn = f"{base_path}/config_historic.json"
config_historic.to_json(file=config_historic_fn)

# check if everything is set correct
check_lpjml(config_historic_fn, model_path)
# submit spinup job and get corresponding id
historic_jobid = submit_lpjml(
    config_file=config_historic_fn, model_path=model_path,
    output_path=output_path, dependency=spinup_jobid
)


# define coupled run -------------------------------------------------------- #

# create config for coupled run
config_coupled = parse_config(path=model_path)
# set coupled run configuration
config_coupled.set_couple(output_path, restart_path, start=1981, end=2005,
                          couple_inputs=["landuse", "fertilizer_nr"],
                          couple_outputs=["cftfrac", "pft_harvestc",
                                          "pft_harvestn"],
                          write_outputs=["prec", "transp", "interc", "evap",
                                         "runoff", "discharge", "fpc", "vegc",
                                         "soilc", "litc", "cftfrac",
                                         "pft_harvestc", "pft_harvestn",
                                         "pft_rharvestc", "pft_rharvestn",
                                         "pet", "leaching"],
                          write_temporal_resolution="annual")

# write config (LpjmlConfig object) as json file
config_coupled_fn = f"{base_path}/config_coupled.json"
config_coupled.to_json(file=config_coupled_fn)

# submit coupled run -------------------------------------------------------- #

# check if everything is set correct
check_lpjml(config_coupled_fn, model_path)
# submit spinup job and get corresponding id
historic_jobid = submit_lpjml(
    config_file=config_coupled_fn, model_path=model_path,
    output_path=output_path, dependency=historic_jobid,
    couple_to="<COPAN:CORE>"
)
