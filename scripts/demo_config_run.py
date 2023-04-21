import os
from pycoupler.utils import check_lpjml, compile_lpjml, clone_lpjml, \
    create_subdirs
from pycoupler.config import read_config
from pycoupler.run import run_lpjml


# paths
sim_path = "/p/projects/open/Jannes/copan_core/lpjml"
model_path = f"{sim_path}/LPJmL_internal"


startcell = 27410
endcell = startcell

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
spinup_path = config_spinup.set_spinup(sim_path)

# only for single cell runs
config_spinup.startgrid = startcell
config_spinup.endgrid = endcell
config_spinup.river_routing = False

# write config (Config object) as json file
config_spinup_fn = config_spinup.to_json(path=sim_path)

# check if everything is set correct
check_lpjml(config_file=config_spinup_fn, model_path=model_path)
# run spinup job
run_lpjml(
    config_file=config_spinup_fn,
    model_path=model_path,
    output_path=spinup_path
)

# define and submit historic run -------------------------------------------- #

# create config for historic run
config_historic = read_config(file_name=f"{model_path}/lpjml.js")

# set historic run configuration
historic_path = config_historic.set_historic(sim_path,
                                             start_year=1901, end_year=1980,
                                             write_start_year=1980)

# only for single cell runs
config_historic.startgrid = startcell
config_historic.endgrid = endcell
config_historic.river_routing = False

# write config (Config object) as json file
config_historic_fn = config_historic.to_json(path=sim_path)


# check if everything is set correct
check_lpjml(config_historic_fn, model_path)
# run spinup job
run_lpjml(
    config_file=config_historic_fn,
    model_path=model_path,
    output_path=historic_path
)


# define coupled run -------------------------------------------------------- #

# create config for coupled run
config_coupled = read_config(file_name=f"{model_path}/lpjml.js")
# set coupled run configuration
coupled_path = config_coupled.set_coupled(sim_path,
                                          start_year=1981, end_year=2005,
                                          couple_inputs=["landuse",
                                                         "fertilizer_nr"],
                                          couple_outputs=["cftfrac",
                                                          "pft_harvestc",
                                                          "pft_harvestn"],
                                          write_outputs=["prec", "transp", 
                                                         "interc", "evap",
                                                         "runoff", "discharge",
                                                         "fpc", "vegc",
                                                         "soilc", "litc",
                                                         "cftfrac",
                                                         "pft_harvestc",
                                                         "pft_harvestn",
                                                         "pft_rharvestc",
                                                         "pft_rharvestn",
                                                         "pet", "leaching"],
                                          write_temporal_resolution=None)

# only for single cell runs
config_coupled.startgrid = startcell
config_coupled.endgrid = endcell
config_coupled.river_routing = False

# write config (Config object) as json file
config_coupled_fn = config_coupled.to_json(path=sim_path)

# submit coupled run -------------------------------------------------------- #

# check if everything is set correct
check_lpjml(config_coupled_fn, model_path)
# run lpjml simulation for coupling
run_lpjml(
    config_file=config_coupled_fn,
    model_path=model_path,
    output_path=coupled_path
)

# --------------------------------------------------------------------------- #
# OPEN SECOND LOGIN NODE
# --------------------------------------------------------------------------- #
from pycoupler.coupler import LPJmLCoupler
from pycoupler.data import supply_inputs, preprocess_inputs


sim_path = "/p/projects/open/Jannes/copan_core/lpjml"
model_path = f"{sim_path}/LPJmL_internal"
config_historic_fn = f"{sim_path}/config_historic.json"
config_coupled_fn = f"{sim_path}/config_coupled.json"
coupler = LPJmLCoupler(config_file=config_coupled_fn)


# get and process initial inputs
inputs = supply_inputs(config_file=config_coupled_fn,
                       historic_config_file=config_historic_fn,
                       input_path=f"{sim_path}/input",
                       model_path=model_path,
                       start_year=1981, end_year=1981)

input_data = preprocess_inputs(inputs, grid=coupler.grid, time=1980)

# coupled simulation years
years = range(1981, 2006)
#  The following could be your model/program/script
for year in years:
    # send input data to lpjml
    coupler.send_inputs(input_data, year)
    # read output data
    outputs = coupler.read_outputs(year)
    # generate some results based on lpjml outputs
    # ....

coupler.close_channel()
