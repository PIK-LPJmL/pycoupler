# pycoupler  <a href=''><img src='docs/img/logo.png' align="right" /></a>

The pycoupler project serves a LPJmL Python interface to be used to operate
[LPJmL](https://gitlab.pik-potsdam.de/lpjml/LPJmL_internal) within a Python
environment and to couple it to Python-based modelling frameworks.  
pycoupler was written with the intention to support the coupling of
[copan:core](https://github.com/pik-copan/pycopancore/) with LPJmL.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pycopler.

```bash
pip install
```

Please make sure to also have set [working environment for LPJmL](https://gitlab.pik-potsdam.de/lpjml/LPJmL_internal/-/blob/master/INSTALL) correctly if you are not working
on the PIK cluster (with Slurm Workload Manager).  
Else please add this to to working environment or profile

```bash
module purge
module load python/3.8.2
module load lpjml

export LPJROOT=<PATH_TO_LOCAL_LPJML_REPOSITORY>/LPJmL_internal
# ONLY if you do not have access to [LPJmL](https://gitlab.pik-potsdam.de/lpjml/LPJmL_internal)
#   for usage of python function clone_lpjml
export GIT_LPJML_URL="gitlab.pik-potsdam.de/lpjml/LPJmL_internal.git"
export GIT_READ_TOKEN="<ASK_THE_AUTHOR_OF_THIS_PACKAGE>"
```

## Usage (example)

```python
from pycoupler.config import parse_config
from pycoupler.utils import check_lpjml
from pycoupler.submit import submit_couple

# paths
model_path = "<LPJROOT>"
output_path ="<DIR_TO_WRITE_LPJML_OUTPUT>"


# define a coupled LPJmL run provided the spinup and historic runs have already
#   been performed 

# create config for coupled run based on lpjml.js in LPJROOT
config_coupled = parse_config(path=model_path)
# set output directory, outputs (relevant ones for pbs and agriculture)
config_coupled.set_outputs(
    output_path,
    outputs=["vegc", "soilc", "litc", "cftfrac", "pft_harvestc"],
    temporal_resolution="annual",
    file_format="cdf"
)
# set coupling parameters
config_coupled.set_coupler(
    inputs=["landuse", "fertilizer_nr", "with_tillage", "residue_on_field"],
    outputs=["cftfrac", "pft_harvestc"]
)

# write config (LpjmlConfig object) as json file
config_coupled_fn = "config_coupled.json"
config_coupled.to_json(file=config_coupled_fn)


# submit coupled run to slurm

# check if everything is set correct
check_lpjml(config_coupled_fn, model_path)
# submit spinup job and get corresponding id
historic_jobid = submit_couple(
    config_file=config_coupled_fn, model_path=model_path,
    output_path=output_path, couple="<COUPLING_PROGRAM_MAIN>.py"
)

```

## Contributing
Merge requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
