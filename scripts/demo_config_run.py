import os
from pycoupler.utils import check_lpjml, compile_lpjml, clone_lpjml, \
    create_subdirs
from pycoupler.config import read_config
from pycoupler.run import run_lpjml
from pycoupler.coupler import LPJmLCoupler
from pycoupler.data import copy_coupled_input, read_coupled_input

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
config_spinup.set_spinup(sim_path)

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

# only for single cell runs
config_coupled.startgrid = startcell
config_coupled.endgrid = endcell
config_coupled.river_routing = False

# write config (Config object) as json file
config_coupled_fn = config_coupled.to_json(path=sim_path)

# run coupled sim ----------------------------------------------------------- #

# check if everything is set correct
check_lpjml(config_coupled_fn, model_path)

# run lpjml simulation for coupling in the background
run_lpjml(
    config_file=config_coupled_fn,
    model_path=model_path,
    sim_path=sim_path
)

# set up coupler and simulation --------------------------------------------- #
coupler = LPJmLCoupler(config_file=config_coupled_fn)

# convert and copy coupled input data
copy_coupled_input(coupler=coupler,
                   sim_path=sim_path,
                   model_path=model_path)

# read coupled input data for initialisation
inputs = read_coupled_input(coupler=coupler,
                            sim_path=sim_path)

# get historic outputs
historic_outputs = coupler.read_historic_output()

# coupled run --------------------------------------------------------------- #
#  The following could be your model/program/script
for year in coupler.get_sim_years():
    # send input data to lpjml
    coupler.send_input(inputs, year)
    # read output data
    outputs = coupler.read_output(year)
    # generate some results based on lpjml output
    # ...

coupler.close()
