from pycoupler.utils import check_lpjml, compile_lpjml, clone_lpjml
from pycoupler.config import parse_config
from pycoupler.run import run_lpjml


# paths
model_location = "<INSERT_MODEL_LOCATION>"
model_path = f"{model_location}/LPJmL_internal"
base_path = "<INSERT_PATH_TO_ENCLOSING_FOLDER_OF_MODEL_OUTPUT_RESTART_INPUT>"
output_path = f"{base_path}/output"
restart_path = f"{base_path}/restart"

cell = 27410

# set up lpjml -------------------------------------------------------------- #

# clone function to model location via oauth token (set as enironment var) and
#   checkout copan branch (default until it is merged)
clone_lpjml(model_location=model_location, branch="lpjml53_copan")
# if patched and existing compiled version use make_fast=True or if error is
#   thrown, use arg make_clean=True without make_fast=True
compile_lpjml(model_path=model_path)

# define and submit spinup run ---------------------------------------------- #

# create config for spinup run
config_spinup = parse_config(path=model_path, spin_up=True)
# set spinup run configuration
config_spinup.set_spinup(output_path, restart_path)
# only for single cell runs
config_spinup.startgrid = cell
config_spinup.river_routing = False
# write config (LpjmlConfig object) as json file
config_spinup_fn = f"{base_path}/config_spinup.json"
config_spinup.to_json(file=config_spinup_fn)

# check if everything is set correct
check_lpjml(config_file=config_spinup_fn, model_path=model_path)
# run spinup job
run_lpjml(
    config_file=config_spinup_fn, model_path=model_path,
    output_path=output_path
)

# define and submit historic run -------------------------------------------- #

# create config for historic run
config_historic = parse_config(path=model_path)
# set historic run configuration
config_historic.set_historic(output_path, restart_path, start=1901, end=1980,
                             write_start=1980)
# only for single cell runs
config_historic.startgrid = cell
config_historic.river_routing = False
# write config (LpjmlConfig object) as json file
config_historic_fn = f"{base_path}/config_historic.json"
config_historic.to_json(file=config_historic_fn)

# check if everything is set correct
check_lpjml(config_historic_fn, model_path)
# run spinup job
run_lpjml(
    config_file=config_historic_fn, model_path=model_path,
    output_path=output_path
)


# define coupled run -------------------------------------------------------- #

# create config for coupled run
config_coupled = parse_config(path=model_path)
# set coupled run configuration
config_coupled.set_couple(output_path, restart_path, start=1981, end=2005,
                          inputs=["landuse", "fertilizer_nr"],
                          outputs=["cftfrac", "pft_harvestc", "pft_harvestn"],
                          write_outputs=["prec", "transp", "interc", "evap",
                                         "runoff", "discharge", "fpc", "vegc",
                                         "soilc", "litc", "cftfrac",
                                         "pft_harvestc", "pft_harvestn",
                                         "pft_rharvestc", "pft_rharvestn",
                                         "pet", "leaching"],
                          write_temporal_resolution="annual")
# only for single cell runs
config_coupled.startgrid = cell
config_coupled.river_routing = False
# write config (LpjmlConfig object) as json file
config_coupled_fn = f"{base_path}/config_coupled.json"
config_coupled.to_json(file=config_coupled_fn)

# submit coupled run -------------------------------------------------------- #

# check if everything is set correct
check_lpjml(config_coupled_fn, model_path)
# run lpjml simulation for coupling
run_lpjml(
    config_file=config_coupled_fn, model_path=model_path,
    output_path=output_path
)

# --------------------------------------------------------------------------- #
# OPEN SECOND LOGIN NODE
# --------------------------------------------------------------------------- #
from pycoupler.coupler import Coupler
from pycoupler.data import supply_inputs, preprocess_inputs


model_location = "<INSERT_MODEL_LOCATION>"
base_path = "<INSERT_PATH_TO_ENCLOSING_FOLDER_OF_MODEL_OUTPUT_RESTART_INPUT>"
model_path = f"{model_location}/LPJmL_internal"
config_historic_fn = f"{base_path}/config_historic.json"
config_coupled_fn = f"{base_path}/config_coupled.json"
coupler = Coupler(config_file=config_coupled_fn)

# get and process initial inputs
inputs = supply_inputs(config_file=config_coupled_fn,
                       historic_config_file=config_historic_fn,
                       input_path=f"{base_path}/input",
                       model_path=model_path,
                       start_year=1981, end_year=1981)
input_data = preprocess_inputs(inputs, grid=coupler.grid, time=1980)

# coupled simulation years
years = range(1981, 2005)
#  The following could be your model/program/script
for year in years:
    # send input data to lpjml
    coupler.send_inputs(input_data, year)
    # read output data
    outputs = coupler.read_outputs(year)
    # generate some results based on lpjml outputs
    # ....

coupler.close_channel()
