__all__ = ["LpjmlConfig", "parse_config", "read_config", "submit_lpjml",
           "compile_lpjml", "check_lpjml", "clone_lpjml", "Coupler",
           "run_lpjml", "supply_inputs"]

import os
from packaging import version

# check if working environment is PIK's cluster (workaround by Ciaron)
#   (might be adjusted to the new cluster coming soon ...)
if os.path.isdir('/p/system'):
    versions = [version.parse(vers) for vers in os.listdir(
        "/p/system/packages/modules/")]
    exec(open(
        f"/p/system/packages/modules/{max(versions)}/init/python.py"
    ).read())
    silent = module('load', 'lpjml')
    os.environ['I_MPI_DAPL_UD'] = 'disable'
    os.environ['I_MPI_FABRICS'] = 'shm:shm'
    os.environ['I_MPI_DAPL_FABRIC'] = 'shm:sh'
else:
    # if you are installing LPJmL somewhere else you are responsible to set
    #   the environment correct
    print("Please make sure to have LPJmL configured right!" +
          "https://gitlab.pik-potsdam.de/lpjml/LPJmL_internal/" +
          "-/blob/master/INSTALL")
