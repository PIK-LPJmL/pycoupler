try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "1.5.0"

from .config import (
    LpjmlConfig,
    CoupledConfig,
    read_config,
    read_yaml,
)

from .coupler import LPJmLCoupler, kill_process_on_port

from .run import run_lpjml, submit_lpjml, check_lpjml, start_lpjml, \
    kill_stale_lpjml_processes

from .data import (
    LPJmLData,
    LPJmLDataSet,
    LPJmLMetaData,
    read_data,
    read_meta,
    read_header,
    get_headersize,
)

from .utils import (
    get_countries,
    search_country,
    read_json,
    detect_io_type,
)

__all__ = [
    "LpjmlConfig",
    "CoupledConfig",
    "read_config",
    "read_yaml",
]

__all__ += ["LPJmLCoupler", "kill_process_on_port"]

__all__ += [
    "LPJmLData",
    "LPJmLDataSet",
    "LPJmLMetaData",
    "read_data",
    "read_meta",
    "read_header",
    "get_headersize",
]

__all__ += ["run_lpjml", "start_lpjml", "submit_lpjml", "check_lpjml", "kill_stale_lpjml_processes"]  # noqa: E501

__all__ += [
    "get_countries",
    "search_country",
    "read_json",
    "detect_io_type",
]

__all__ += ["__version__"]
