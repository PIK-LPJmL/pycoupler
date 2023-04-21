__all__ = ["LpjmlConfig", "parse_config", "read_config", "submit_lpjml",
           "compile_lpjml", "check_lpjml", "clone_lpjml", "LPJmLCoupler",
           "run_lpjml", "supply_inputs"]

import os
import warnings
from packaging import version

# check if working environment is PIK's cluster (workaround by Ciaron)
#   (might be adjusted to the new cluster coming soon ...)
if os.path.isdir('/p/system'):
    # initiate loading the clusters environment modules from within python
    #   https://modules.readthedocs.io/en/latest/module.html?highlight=python
    versions = [version.parse(vers) for vers in os.listdir(
        "/p/system/packages/modules/")]
    exec(open(
        f"/p/system/packages/modules/{max(versions)}/init/python.py"
    ).read())
    # load lpjml module
    module('load', 'lpjml')

else:
    # if you are installing LPJmL somewhere else you are responsible to set
    #   the environment correct
    print("Please make sure to have LPJmL configured right!" +
          "https://gitlab.pik-potsdam.de/lpjml/LPJmL_internal/" +
          "-/blob/master/INSTALL")

if not os.environ.get('LPJROOT'):
    warnings.warn(("Environment variable 'LPJROOT' is not set. If you want" +
                   " to use LPJmL functions set 'LPJROOT' first (see also" +
                   " README.md)"))
