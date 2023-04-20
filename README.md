# pycoupler  <a href=''><img src='docs/img/logo.png' align="right" height="139" /></a>

The pycoupler project serves a LPJmL Python interface to be used to operate
[LPJmL](https://gitlab.pik-potsdam.de/lpjml/LPJmL_internal) within a Python
environment and to couple it to Python-based modelling frameworks.  
pycoupler was written with the intention to support the coupling of
[copan:core](https://github.com/pik-copan/pycopancore/) with LPJmL.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pycoupler.

```bash
pip install .
```

Please make to clone LPJmL in advance and provide its path via `model_path`.  
Also make sure to also have set the [working environment for LPJmL](https://gitlab.pik-potsdam.de/lpjml/LPJmL_internal/-/blob/master/INSTALL) correctly if you are not working
on the PIK cluster (with Slurm Workload Manager).  


## Usage

##### **1. terminal: run LPJmL**
```python
from pycoupler.config import parse_config
from pycoupler.utils import check_lpjml
from pycoupler.run import run_lpjml


# paths
model_path = "<model directory>"
output_path ="<output writing directory>"

# define a coupled LPJmL run provided the spinup and historic runs have already
#   been performed (else look into ./scripts/demo_config_run.py)

# create config for coupled run based on lpjml.js in LPJROOT
config_coupled = parse_config(path=model_path)

# set coupled run configuration
config_coupled.set_couple(output_path, restart_path, start=2023, end=2032,
                          couple_inputs=["landuse", "fertilizer_nr"],
                          couple_outputs=["cftfrac", "pft_harvestc",
                                          "pft_harvestn"],
                          write_outputs=["prec", "transp", "interc", "evap",
                                         "runoff", "discharge", "fpc", "vegc",
                                         "soilc", "litc", "pft_rharvestc",
                                         "pft_rharvestn", "pet", "leaching"],
                          write_temporal_resolution="annual")

# only for single cell runs
config_coupled.startgrid = 27410
config_coupled.river_routing = False

# write config (LpjmlConfig object) as json file
config_coupled_fn = "./config_coupled.json"
config_coupled.to_json(file=config_coupled_fn)

# check if everything is set correct
check_lpjml(config_coupled_fn, model_path)

# run lpjml simulation for coupling
run_lpjml(
    config_file=config_coupled_fn, model_path=model_path,
    output_path=output_path
)
```

##### **2. terminal: your program (model, script, ...)**

```python
from pycoupler.coupler import Coupler
from pycoupler.data import supply_inputs, preprocess_inputs


# paths
input_path = "<input data directory>"
config_coupled_fn = "./config_coupled.json"

# coupled simulation years
start_year = 2023
end_year = 2032
sim_years = range(start_year, end_year+1)

# initiate coupler after run_lpjml on LOGIN NODE 1
coupler = Coupler(config_file=config_coupled_fn)

# get and process initial inputs
inputs = supply_inputs(config_file=config_coupled_fn,
                       historic_config_file=config_historic_fn,
                       input_path=input_path,
                       model_path=model_path,
                       start_year=start_year,
                       end_year=start_year)
input_data = preprocess_inputs(inputs, grid=coupler.grid, time=start_year)

# coupled simulation years
sim_years = range(start_year, end_year+1)
#  The following could be your model/program/script
for year in sim_years:

    # send input data to lpjml
    coupler.send_inputs(input_data, year)

    # read output data
    outputs = coupler.read_outputs(year)

    # generate some results based on lpjml outputs
    # ....

coupler.close_channel()

```

## Contributing
Merge requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate.
