import os
import struct
import re
from pathlib import Path
from collections.abc import Hashable
from typing import Dict
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
from xarray.core.variable import Variable
import xarray.core.utils as utils

from pycoupler.utils import read_json

DEFAULT_NETCDF_FILL_VALUE = np.float32(-999.0)
DEFAULT_NETCDF_FILL_VALUE_INT = -9999


def _suppress_coordinate_fill(dataset: xr.Dataset) -> None:
    """Remove _FillValue metadata from coordinate variables."""

    for coord_name in dataset.coords:
        coord = dataset[coord_name]
        coord.attrs.pop("_FillValue", None)
        coord.encoding["_FillValue"] = None


def _sanitize_prefix(value: str | None, default: str = "lpjml") -> str:
    """Return a filesystem-friendly prefix."""
    candidate = (value or "").strip()
    if not candidate:
        candidate = default
    safe = re.sub(r"[^\w\-]+", "_", candidate).strip("_")
    return safe or default


class LPJmLInputType:
    """Available Input types loaded from config.

    Parameters
    ----------
    id : int, optional
        ID of the input type.
    name : str, optional
        Name of the input type.

    Attributes
    ----------
    id : int
        ID of the input type.
    name : str
        Name of the input type.
    nband : int
    """

    __input_types__ = None  # This will hold the configuration data

    def __init__(self, id=None, name=None):
        """Initialize the instance with an id (index)."""
        if not LPJmLInputType.__input_types__:
            raise RuntimeError(
                "LPJmLInputType has not been initialized. Call 'load_config' first."
            )

        if id is not None:
            # Find the corresponding input id from the provided id
            self.__dict__.update(
                next(
                    value.update({"name": key}) or value
                    for key, value in LPJmLInputType.__input_types__.items()
                    if value["id"] == id
                )
            )
        elif name is not None:
            # Find the corresponding input name from the provided name
            self.__dict__.update(
                next(
                    value.update({"name": key}) or value
                    for key, value in LPJmLInputType.__input_types__.items()
                    if key == name
                )
            )
        else:
            raise ValueError("Either 'id' or 'name' must be provided.")

    @classmethod
    def load_config(cls, config):
        """Load input types from the provided config."""
        cls.__input_types__ = config.input.to_dict()
        # Filter to entries with "id" (proper input types); skip scalars like
        # delta_year
        cls.__input_types__ = {
            k: v
            for k, v in cls.__input_types__.items()
            if isinstance(v, dict) and "id" in v
        }
        cls.names = list(cls.__input_types__.keys())
        cls.ids = [value["id"] for value in cls.__input_types__.values()]

    @property
    def nband(self):
        """Get amount of bands"""
        if self.name == "landuse":
            return 64
        if self.name in ["fertilizer_nr", "manure_nr"]:
            return 32
        if self.name in ["sdate", "crop_phu"]:
            return 24
        if self.name == "residue_on_field":
            return 16
        return 1

    @property
    def type(self):
        """Get the data type for the specific input"""
        if self.name in ["with_tillage", "sdate", "cover_crop"]:
            return int
        else:
            return float

    @property
    def has_bands(self):
        """Check if multiple bands exist (better check for categorical
        bands)"""
        return bool(self.nband > 1)


def append_to_dict(data_dict, data):
    """
    Append data along the third dimension to the data_dict.

    Parameters
    ----------
    data_dict : dict
        Dictionary holding the data.
    data : dict
        Dictionary with data.
        Keys are ids/names, values are two-dimensional NumPy arrays with
        dimensions (cells, bands).

    Returns
    -------
    dict
        Updated data_dict with the appended data.
    """
    for key, value in data.items():
        if key in data_dict:
            data_dict[key] = np.dstack((data_dict[key], value))

        else:
            data_dict[key] = value

    return data_dict


def _ensure_cf_metadata(
    data_array: xr.DataArray,
) -> Dict[str, str] | None:
    """Attach CF-compliant coordinate metadata and extract globals."""

    da = data_array

    def _ensure(coord_name: str, attrs: Dict[str, str]):
        if coord_name in da.coords:
            coord = da.coords[coord_name]
            for key, value in attrs.items():
                coord.attrs.setdefault(key, value)

    _ensure(
        "lat",
        {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
    )
    _ensure(
        "lon",
        {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    )
    if "time" in da.coords:
        coord = da.coords["time"]
        coord.attrs.setdefault("standard_name", "time")
        coord.attrs.setdefault("long_name", "Time")
        coord.attrs.setdefault("axis", "T")
        coord.attrs.setdefault("units", "years")
        coord.attrs.setdefault("calendar", "noleap")

    _ensure(
        "area_km2",
        {"long_name": "grid cell area", "units": "km2"},
    )

    if not da.attrs.get("units"):
        da.attrs["units"] = "1"

    return da.attrs.pop("_global_attrs", None)


class LPJmLData(xr.DataArray):
    """Class for single LPJmL data arrays (input, output, etc.) with meta data
    and defined dimensions (cell, band, time).

    Parameters
    ----------
    *args : tuple
        Arguments for the xarray.DataArray constructor.
    **kwargs : dict
        Keyword arguments for the xarray.DataArray constructor.

    Attributes
    ----------
    attrs : dict
        Attributes of the data array.
    coords : dict
        Coordinates of the data array.
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLData, self).__init__(*args, **kwargs)

    def add_meta(self, meta_data):
        """
        Add meta data to the data array.

        Parameters
        ----------
        meta_data : LPJmLMetaData
            Meta data to be added to the data array.
        """
        if isinstance(meta_data, LPJmLMetaData):
            self.attrs["standard_name"] = meta_data.variable
            self.attrs["long_name"] = meta_data.long_name
            self.attrs["units"] = meta_data.unit
            self.attrs["source"] = meta_data.source
            self.attrs["history"] = meta_data.history

            if meta_data.cellsize_lat != meta_data.cellsize_lon:
                raise ValueError(
                    "Cell sizes in latitude and longitude direction must be "
                    "equal!"  # noqa: E501
                )
            else:
                self.attrs["cellsize"] = meta_data.cellsize_lon

            band_dim = next(
                (dim for dim in self.dims if dim.startswith("band")), None
            )  # noqa: E501
            # TODO assign lat lon to grid object
            if band_dim is not None and len(self.coords[band_dim]) > 1:
                if meta_data.variable == "grid":
                    self.coords[band_dim] = ["lon", "lat"]
                elif len(self.coords[band_dim]) == len(meta_data.band_names):
                    self.coords[band_dim] = meta_data.band_names
                else:
                    self.coords[band_dim] = np.arange(
                        1, len(self.coords[band_dim]) + 1
                    )  # noqa: E501

            if hasattr(meta_data, "global_attrs"):
                self.attrs["institution"] = meta_data.global_attrs[
                    "institution"
                ]  # noqa: E501
                self.attrs["contact"] = meta_data.global_attrs["contact"]
                self.attrs["comment"] = meta_data.global_attrs["comment"]
        else:
            raise TypeError("meta_data must be of type LPJmLMetaData!")

    def get_neighbourhood(self, id=True, cellsize=0.5):
        """
        Get the IDs of all neighbouring cells within a given size of cells.

        Parameters
        ----------
        id : bool, default True
            If True, return cell ids, else return cell indices
        cellsize : float, default 0.5
            Size of cells in degrees.

        Returns
        -------
        :return: Array with the IDs of all neighbouring cells.
        :rtype: numpy.ndarray
        """

        # Get the coordinates of all cells
        coords_np = np.array([self.cell.lat.values, self.cell.lon.values]).T

        # Build a KDTree for fast nearest-neighbour lookup
        tree = KDTree(coords_np)

        if "cellsize" in self.attrs:
            cellsize = self.cellsize  # in degrees

        # Find all neighbours within the given size of cells
        neighbour_indices = tree.query_ball_point(coords_np, r=cellsize * 1.5)

        # Initialize the array to hold the neighbour cell IDs
        max_neighbours = 8
        n_cells = len(self.cell)
        neighbour_ids = np.full((n_cells, max_neighbours), -9999, dtype=int)

        # Loop over all cells and find their neighbours
        cell_indices = self.cell.values
        for i in range(n_cells):
            # Get the indices of all neighbours for this cell
            current_neighbours = neighbour_indices[i]

            # Remove the current cell from the list of neighbours
            current_neighbours = [n for n in current_neighbours if n != i]

            # Truncate the list to at most max_neighbours
            current_neighbours = current_neighbours[:max_neighbours]

            # Store the neighbour cell IDs in the output array
            if len(current_neighbours) > 0 and id:
                neighbour_ids[i, : len(current_neighbours)] = cell_indices[
                    current_neighbours
                ]
            elif len(current_neighbours) > 0 and not id:
                neighbour_ids[i, : len(current_neighbours)] = (
                    current_neighbours  # noqa: E501
                )

        neighbours = LPJmLData(
            data=neighbour_ids,
            dims=("cell", "neighbour"),
            coords=dict(
                cell=self.cell.values,
                lon=(["cell"], self.lon.values),
                lat=(["cell"], self.lat.values),
                neighbour=np.arange(8),
            ),
            name="neighbourhood",
        )

        return neighbours

    def _wrap(self, data_array):
        """Return a LPJmLData instance from a generic DataArray."""
        if isinstance(data_array, LPJmLData):
            return data_array
        if isinstance(data_array, xr.DataArray):
            return self.__class__(data_array)
        raise TypeError(f"Expected xarray.DataArray, got {type(data_array)!r}")

    def transform(self, to="lon_lat"):
        """Transform the spatial layout between cell and lon/lat grid.

        Parameters
        ----------
        to : {"lon_lat", "cell"}, default "lon_lat"
            Target representation. ``"lon_lat"`` reshapes the data onto a
            two-dimensional latitude/longitude grid using the embedded
            coordinates; ``"cell"`` collapses a lon/lat grid back to the
            original one-dimensional ``cell`` dimension.

        Returns
        -------
        LPJmLData
            Transformed data array.
        """

        to = (to or "").lower()
        if to not in {"lon_lat", "cell"}:
            raise ValueError(
                f"Unsupported transform target '{to}'. "
                "Use either 'lon_lat' or 'cell'."
            )

        if to == "lon_lat":
            if "cell" not in self.dims:
                return self.copy()
            if not {"lon", "lat"} <= set(self.coords):
                raise ValueError(
                    "Cannot transform to lon/lat grid without 'lon' and 'lat' "
                    "coordinates attached to the 'cell' dimension."
                )

            # Ensure lat/lon combinations are unique before unstacking
            index = pd.MultiIndex.from_arrays(
                [self.coords["lat"].values, self.coords["lon"].values],
                names=("lat", "lon"),
            )
            if not index.is_unique:
                raise ValueError(
                    "Duplicate lat/lon pairs detected; cannot build a unique grid."  # noqa: E501
                )

            lon_lat = self.set_index(cell=("lat", "lon")).unstack("cell")
            # Sort coordinates for consistent orientation (lon ascending,
            # lat descending)
            if "lon" in lon_lat.dims:
                lon_lat = lon_lat.sortby("lon")
            if "lat" in lon_lat.dims:
                lon_lat = lon_lat.sortby("lat", ascending=False)

            dims = list(lon_lat.dims)
            ordered_dims = [dim for dim in ("lat", "lon") if dim in dims]
            ordered_dims.extend(dim for dim in dims if dim not in ordered_dims)
            lon_lat = lon_lat.transpose(*ordered_dims)
            return self._wrap(lon_lat)

        # to == "cell"
        if not {"lon", "lat"} <= set(self.dims):
            return self.copy()

        cell_da = self.sortby("lon").sortby("lat", ascending=False)
        cell_da = cell_da.stack(cell=("lat", "lon")).reset_index("cell")
        # Ensure lon/lat remain attached to the cell dimension
        if "lon" in cell_da.coords:
            cell_da.coords["lon"] = ("cell", cell_da.coords["lon"].values)
        if "lat" in cell_da.coords:
            cell_da.coords["lat"] = ("cell", cell_da.coords["lat"].values)

        dims = list(cell_da.dims)
        ordered_dims = ["cell"]
        ordered_dims.extend(dim for dim in dims if dim != "cell")
        cell_da = cell_da.transpose(*ordered_dims)
        return self._wrap(cell_da)

    def to_netcdf(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        compression: bool = True,
        complevel: int = 4,
        fill_value: float | None = None,
        engine: str = "netcdf4",
        mode: str = "w",
        compute: bool = True,
        **kwargs,
    ):
        """Write this LPJmLData to NetCDF, analogous to
        :meth:`xarray.DataArray.to_netcdf`, but gridding ``cell`` dimensions
        before persisting. [#da_netcdf]_

        .. [#da_netcdf] https://docs.xarray.dev/en/latest/generated/xarray.DataArray.to_netcdf.html # noqa: E501
        """

        kwargs = dict(kwargs)
        kwargs.pop("lpjml_style", None)

        lpjml = self
        if lpjml.name is None and path is None:
            raise ValueError(
                "LPJmLData must have a name when no output path is provided."
            )

        if "cell" in lpjml.dims:
            if not {"lon", "lat"} <= set(lpjml.coords):
                raise ValueError(
                    "Cannot write LPJmLData with a 'cell' dimension that lacks "  # noqa: E501
                    "'lon' and 'lat' coordinates."
                )
            lpjml = lpjml.transform("lon_lat")

        global_attrs = _ensure_cf_metadata(lpjml)
        var_name = lpjml.name or "__lpjml_data__"
        dataset = lpjml.to_dataset(name=var_name)
        _suppress_coordinate_fill(dataset)
        if global_attrs:
            dataset.attrs.update(dict(global_attrs))
        dataset.attrs.setdefault("Conventions", "CF-1.8")
        dtype = lpjml.dtype
        encoding: dict[str, dict[str, object]] = {}

        enc: dict[str, object] = {}
        target_fill = fill_value
        if target_fill is None:
            if np.issubdtype(dtype, np.floating):
                target_fill = DEFAULT_NETCDF_FILL_VALUE
            elif np.issubdtype(dtype, np.integer):
                target_fill = DEFAULT_NETCDF_FILL_VALUE_INT
        if target_fill is not None:
            enc["_FillValue"] = target_fill

        if compression and np.issubdtype(dtype, np.number):
            enc["zlib"] = True
            enc["complevel"] = complevel

        encoding[var_name] = enc

        if path is None:
            return dataset.to_netcdf(
                path=None,
                engine=engine,
                mode=mode,
                encoding=encoding,
                compute=compute,
                **kwargs,
            )

        target_path = Path(path)
        if not target_path.suffix:
            target_path = target_path.with_suffix(".nc4")

        dataset.to_netcdf(
            target_path,
            engine=engine,
            mode=mode,
            encoding=encoding,
            compute=compute,
            **kwargs,
        )
        return str(target_path)


class LPJmLDataSet(xr.Dataset):
    """Class for LPJmL data sets.

    Parameters
    ----------
    *args : tuple
        Arguments for the xarray.Dataset constructor.
    **kwargs : dict
        Keyword arguments for the xarray.Dataset constructor.

    Attributes
    ----------
    data_vars : dict
        Data variables of the dataset.
    attrs : dict
        Attributes of the dataset.
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLDataSet, self).__init__(*args, **kwargs)

        if self.data_vars and (
            "cellsize" in self[list(self.data_vars)[0]].attrs
        ):  # noqa: E501
            first_attrs = self[list(self.data_vars)[0]].attrs
            self.attrs["source"] = first_attrs["source"]
            self.attrs["history"] = first_attrs["history"]
            self.attrs["cellsize"] = first_attrs["cellsize"]

            if "institution" in first_attrs:
                self.attrs["institution"] = first_attrs["institution"]
                self.attrs["contact"] = first_attrs["contact"]
                self.attrs["comment"] = first_attrs["comment"]

    def to_numpy(self):
        """Return data as numpy array.

        Returns
        -------
        dict
            Dictionary with data variables as keys and corresponding numpy
            arrays as values.
        """
        return {key: value.to_numpy() for key, value in self.data_vars.items()}

    def __setitem__(self, key, value):

        if utils.is_dict_like(key):
            # check for consistency and convert value to dataset
            # loop over dataset variables and set new values
            processed = []
            for name, var in self.items():
                try:
                    var[key] = value[name]
                    processed.append(name)
                except Exception as e:
                    if processed:
                        raise RuntimeError(
                            "An error occurred while setting values of the"
                            f" variable '{name}'. The following variables have"
                            f" been successfully updated:\n{processed}"
                        ) from e
                    else:
                        raise e
        else:
            super().__setitem__(key, value)

    def _construct_dataarray(self, name: Hashable) -> LPJmLData:
        """Construct a LPJmLData by indexing this dataset (Overwritten from
        xarray.Dataset).
        """

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = xr.core.dataset._get_virtual_variable(
                self._variables, name, self.dims
            )
        else:
            # Work on a shallow copy so we don't mutate dataset variables
            variable = variable.copy(deep=False)

        needed_dims = set(variable.dims)
        stripped_dims = {
            re.sub(r"\s*\(.*?\)", "", item) for item in needed_dims
        }  # noqa: E501

        coords: dict[Hashable, Variable] = {}
        # preserve ordering
        for k in self._variables:
            if k in self._coord_names and (
                set(self.variables[k].dims) <= needed_dims
                or set(self.variables[k].dims) <= stripped_dims
            ):
                coords[k] = self.variables[k].copy(deep=False)

        indexes = xr.core.indexes.filter_indexes_from_coords(
            self._indexes, set(coords)
        )  # noqa: E501
        # Copy indexes to avoid mutating dataset-level state
        indexes = {
            key: (idx.copy() if hasattr(idx, "copy") else idx)
            for key, idx in indexes.items()
        }

        # TODO: this is a hack to get around the fact that we don't have
        #  a proper way to represent band dimensions in xarray

        # get the corresponding band dimension
        band_dim = [
            dim
            for dim in variable._dims
            if dim.startswith("band") and dim != "band"  # noqa: E501
        ]
        if band_dim:
            variable._dims = variable._parse_dimensions(
                [
                    dim if not dim.startswith("band") else "band"
                    for dim in variable._dims
                ]
            )

        # get the corresponding "band" index and delete all other band indexes
        band_idx = [
            key for key in coords if key.startswith("band") and key != "band"
        ]  # noqa: E501
        if band_idx:
            for key in band_idx:
                if name not in key:
                    del indexes[key]
                else:
                    band_idx_name = key
            indexes["band"] = indexes.pop(band_idx_name)
            indexes["band"].index.name = "band"
            indexes["band"].index.dim = "band"
            indexes["band"].dim = "band"

        # get the corresponding "band" coords and delete all other band coords
        band_coords = [
            key for key in coords if key.startswith("band") and key != "band"
        ]
        if band_coords:
            for key in band_coords:
                if name not in key:
                    del coords[key]
                else:
                    band_coords_name = key
            coords["band"] = coords.pop(band_coords_name)
            coords["band"]._dims = ("band",)

        # rename the band dimension to only name "band"
        if name.startswith("band") and name != "band":
            name = "band"

        return LPJmLData(
            variable, coords, name=name, indexes=indexes, fastpath=True
        )  # noqa: E501

    def to_dict(self, data="list", encoding=False):
        """
        Convert this dataset to a dictionary following xarray naming
        conventions.

        Converts all variables and attributes to native Python objects.
        Useful for converting to JSON. To avoid datetime incompatibility,
        use the ``decode_times=False`` argument in ``xarray.open_dataset``.

        Parameters
        ----------
        data : bool or {"list", "array", "lpjmldata"}, optional, default:
        "list"
            Whether to include the actual data in the dictionary.
            - If set to ``False``, returns just the schema.
            - If set to ``"array"``, returns data as the underlying array type.
            - If set to ``"list"`` (or ``True`` for backwards compatibility),
              returns data in lists of Python data types. For efficient "list"
              output, use ``ds.compute().to_dict(data="list")``.
        encoding : bool, optional, default: False
            Whether to include the Dataset's encoding in the dictionary.

        Returns
        -------
        dict
            A dictionary with keys: ``"coords"``, ``"attrs"``, ``"dims"``,
            ``"data_vars"``, and optionally ``"encoding"``.

        See Also
        --------
        xarray.Dataset.from_dict
        xarray.DataArray.to_dict
        """
        if data == "lpjmldata":
            return {var_name: self[var_name] for var_name in self.data_vars}

        return super().to_dict(data=data, encoding=encoding)

    def transform(self, to="lon_lat"):
        """Transform all LPJmLData variables to the requested spatial
        layout."""

        transformed = {}
        for name, data in self.data_vars.items():
            if isinstance(data, LPJmLData):
                transformed[name] = data.transform(to=to)
            else:
                transformed[name] = data

        result = LPJmLDataSet(transformed)
        result.attrs = self.attrs.copy()
        return result

    def to_netcdf(
        self,
        path: str | os.PathLike[str],
        *,
        lpjml_style: bool = True,
        per_variable: bool = True,
        file_prefix: str | None = None,
        suffix: str = ".nc4",
        compression: bool = True,
        complevel: int = 4,
        fill_value: float | None = None,
        engine: str | None = None,
        mode: str = "w",
        compute: bool = True,
        **kwargs,
    ) -> str | Dict[str, str]:
        """Write the dataset to NetCDF.

        By default LPJmL conventions are applied: every variable is aligned on
        the LPJmL grid and written to a dedicated ``<prefix>_<var>.nc4`` file
        (``per_variable=True``). Set ``per_variable=False`` to emit a single
        multi-variable file. Pass ``lpjml_style=False`` to defer entirely to
        the underlying :meth:`xarray.Dataset.to_netcdf`.
        """

        kwargs = dict(kwargs)
        per_variable = kwargs.pop("split_vars", per_variable)
        kwargs.pop("file_prefix", None)
        kwargs.pop("suffix", None)

        if not lpjml_style:
            return super().to_netcdf(
                path=path,
                engine=engine,
                mode=mode,
                compute=compute,
                **kwargs,
            )

        aligned = _grid_aligned_dataset(self)
        _suppress_coordinate_fill(aligned)
        if per_variable:
            target_dir = Path(path)
            target_dir.mkdir(parents=True, exist_ok=True)
            prefix_default = (
                file_prefix
                or aligned.attrs.get("sim_name")
                or target_dir.name
                or "lpjml"
            )
            prefix = _sanitize_prefix(prefix_default)
            written: Dict[str, str] = {}
            for name, data in aligned.data_vars.items():
                target_file = target_dir / f"{prefix}_{name}{suffix}"
                LPJmLData(data).to_netcdf(
                    target_file,
                    compression=compression,
                    complevel=complevel,
                    fill_value=fill_value,
                    engine=engine or "netcdf4",
                    mode=mode,
                    compute=compute,
                    **kwargs,
                )
                written[name] = str(target_file)
            return written

        encoding = {
            name: _netcdf_encoding(
                data,
                fill_value=fill_value,
                compression=compression,
                complevel=complevel,
            )
            for name, data in aligned.data_vars.items()
        }
        target_path = os.fspath(path)
        aligned.to_netcdf(
            target_path,
            engine=engine or "netcdf4",
            mode=mode,
            encoding=encoding,
            compute=compute,
            **kwargs,
        )
        return target_path


def read_data(file_name, var_name=None, multiple_bands=False):
    """Read netcdf file and return data as numpy array or xarray.DataArray.

    Parameters
    ----------
    file_name : str
        Path to netcdf file
    var_name : str, optional
        Name of variable to be read

    Returns
    -------
    LPJmLData
        Data as LPJmLData (xarray.DataArray)
    """
    with xr.open_dataset(
        file_name, decode_times=False, mask_and_scale=False
    ) as data:  # noqa
        if "time" in data.dims:
            unit, reference_date = data.time.attrs["units"].split("since")
            date_time = pd.date_range(
                start=reference_date, periods=data.sizes["time"], freq="MS"
            )
            data.coords["time"].attrs["units"] = unit
            data.coords["time"] = date_time.year

        other_dims = [
            dim for dim in data.dims if dim not in ["lat", "lon", "time"]
        ]  # noqa: E501

        # handle multiple bands
        if var_name and multiple_bands:
            band_dim = f"band ({var_name})"
        else:
            band_dim = "band"

        if other_dims:
            if len(other_dims) == 1:
                data = data.rename({other_dims[0]: band_dim})
            elif len(other_dims) > 1:
                raise ValueError("Unexpected multiple dimensions in data.")

        else:
            data = data.expand_dims(band_dim).assign_coords({band_dim: [0]})

        data.coords[band_dim] = np.arange(data.sizes[band_dim])

        if var_name:
            data = data[var_name]
            data = LPJmLData(data)
        else:
            data = LPJmLDataSet(data)

        return data


class LPJmLMetaData:
    """LPJmL meta data class that can be easily accessed,
    converted to a dictionary or written as a json file.

    Parameters
    ----------
    config_dict : dict
        Takes a dictionary (ideally LPJmL config dictionary) and builds up a
        nested LpjmLConfig class with corresponding fields

    Attributes
    ----------
    sim_name : str
        Name of the simulation.
    source : str
        Source of the data.
    history : str
        History of the data.
    variable : str
        Variable of the data.
    long_name : str
        Long name of the data.
    unit : str
        Unit of the data.
    nbands : int
        Number of bands.
    band_names : list
        Names of the bands.
    nyear : int
        Number of years.
    firstyear : int
        First year of the data.
    lastyear : int
        Last year of the data.
    cellsize_lon : float
        Cell size in longitude.
    cellsize_lat : float
        Cell size in latitude.
    ncell : int
        Number of cells.
    firstcell : int
        First cell of the data.
    datatype : int
        Data type.
    scalar : float
        Scalar value.
    nstep : int
        Number of steps.
    timestep : int
        Time step.
    """

    def __init__(self, meta_dict):
        """Constructor method"""
        self.__dict__.update(meta_dict)
        if "band_names" not in self.__dict__ and hasattr(self, "nbands"):
            self.band_names = [str(ii) for ii in range(self.nbands)]

    def to_dict(self):
        """Convert class object to dictionary

        Returns
        -------
        dict
            Dictionary with the meta data.
        """

        def obj_to_dict(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            result = {}
            for key, val in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                element = []
                if isinstance(val, list):
                    for item in val:
                        element.append(obj_to_dict(item))
                else:
                    element = obj_to_dict(val)
                result[key] = element
            return result

        return obj_to_dict(self)

    def __repr__(self, sub_repr=False):
        """Representation of the LPJmL Meta object"""
        summary_attr = [
            "sim_name",
            "source",
            "history",
            "variable",
            "long_name",
            "unit",
            "nbands",
            "band_names",
            "nyear",
            "firstyear",
            "lastyear",
            "cellsize_lon",
            "cellsize_lat",
            "ncell",
            "firstcell",
            "datatype",
            "scalar",
        ]
        other_attr = [
            to_repr for to_repr in self.__dict__ if to_repr not in summary_attr
        ]
        if sub_repr:
            spacing = "\n  "
            summary = "Meta data:"
        else:
            summary = f"<pycoupler.{self.__class__.__name__}>"
            spacing = "\n"
        # TODO: add solution for sub dicts else error in repr
        summary = spacing.join(
            [
                summary,
                f"  * sim_name      {self.sim_name}",
                f"  * source        {self.source}",
                f"  * history       {self.history}",
                f"  * variable      {self.variable}",
                f"  * long_name     {self.long_name}",
                f"  * unit          {self.unit}",
                f"  * nbands        {self.nbands}",
                f"  * band_names    {self.band_names}",
                f"  * nyear         {self.nyear}",
                f"  * firstyear     {self.firstyear}",
                f"  * lastyear      {self.lastyear}",
                f"  * cellsize_lon  {self.cellsize_lon}",
                f"  * cellsize_lat  {self.cellsize_lat}",
                f"  * ncell         {self.ncell}",
                f"  * firstcell     {self.firstcell}",
                f"  * datatype      {self.datatype}",
                f"  * scalar        {self.scalar}",
            ]
        )
        if other_attr:
            summary_list = [summary]
            summary_list.extend(
                [
                    f"  * {torepr}{(13-len(torepr))*' '} {getattr(self, torepr)}"  # noqa: E501
                    for torepr in other_attr
                ]
            )
            summary = spacing.join(summary_list)
        return summary


def read_meta(file_name):
    """Read meta data from json file and return as LPJmLMetaData object.

    Parameters
    ----------
    file_name : str
        Path to json file

    Returns
    -------
    LPJmLMetaData
        Meta data as LPJmLMetaData object
    """
    return LPJmLMetaData(read_json(file_name))


def _netcdf_encoding(
    data: xr.DataArray,
    *,
    fill_value: float | None,
    compression: bool,
    complevel: int,
) -> Dict[str, object]:
    dtype = data.dtype
    encoding: Dict[str, object] = {}

    if fill_value is not None:
        encoding["_FillValue"] = fill_value
    else:
        if np.issubdtype(dtype, np.floating):
            encoding["_FillValue"] = DEFAULT_NETCDF_FILL_VALUE
        elif np.issubdtype(dtype, np.integer):
            encoding["_FillValue"] = DEFAULT_NETCDF_FILL_VALUE_INT

    if compression and np.issubdtype(dtype, np.number):
        encoding["zlib"] = True
        encoding["complevel"] = complevel

    return encoding


def _grid_aligned_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Return dataset where all cell variables are converted to lon/lat
    grids."""

    data_vars = {}
    for name, data in ds.data_vars.items():
        arr = data if isinstance(data, LPJmLData) else LPJmLData(data)
        if "cell" in arr.dims:
            if not {"lon", "lat"} <= set(arr.coords):
                raise ValueError(
                    f"Variable '{name}' is indexed by 'cell' but lacks "
                    "corresponding 'lon'/'lat' coordinates."
                )
            arr = arr.transform("lon_lat")
        data_vars[name] = arr

    aligned = xr.Dataset(data_vars)
    aligned.attrs.update(ds.attrs)
    aligned.attrs.setdefault("Conventions", "CF-1.8")
    return aligned


# Function has been derived from the lpjmlkit R package
#   https://github.com/PIK-LPJmL/lpjmlkit
#   Author of original R function: Sebastian Ostberg
def read_header(filename, to_dict=False, force_version=None, verbose=False):
    """
    Read header (any version) from LPJmL input/output file

    Reads a header from an LPJmL clm file. CLM is the default format used for
    LPJmL input files and can also be used for output files.

    Parameters
    ----------
    filename : str
        Filename to read header from.
    to_dict : bool, optional
        If True, return header as dictionary. If False, return as
        LPJmLMetaData object.
    force_version : int, optional
        Manually set clm version. The default value `NULL` means that the
        version is determined automatically from the header. Set only if the
        version number in the file header is incorrect.
    verbose : bool, optional
        If `TRUE` (the default), `read_header` provides some
        feedback when using default values for missing parameters. If `FALSE`,
        only errors are reported.

    Returns
    -------
    LPJmLMetaData or dict
        A LPJmLMetaData object or a dictionary with the header information.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    with open(filename, "rb") as f:
        # Read the first 30 bytes to determine the header name
        headername_raw = f.read(30)
        headername = "".join(chr(b) for b in headername_raw)
        first_non_alnum_index = next(
            (
                i
                for i, char in enumerate(headername)
                if not (char.isalnum() or char == "_")
            ),
            len(headername),  # noqa
        )
        headername = headername[:first_non_alnum_index]

        for i, char in enumerate(headername):
            if not (char.isalnum() or char == "_"):
                headername = headername[:i]
                break

        if not headername.startswith("LPJ"):
            raise ValueError(f"Invalid header name {headername}")
        if headername == "LPJRESTART":
            raise ValueError(
                "LPJRESTART header detected. This function does not support"
                + " restart headers at the moment."
            )

        # Skip over the header
        f.seek(len(headername))

        # Determine file endian. Try platform-specific endian as default.
        endian = "little"
        version = struct.unpack(f"<i", f.read(4))[0]

        if version & 0xFF == 0:
            endian = "big"
            f.seek(len(headername))
            version = struct.unpack(f">i", f.read(4))[0]

        if force_version is not None and force_version != version:
            if verbose:
                print(f"Forcing header version to {force_version}")
            version = force_version

        # Read main header attributes that are included in all header versions
        if endian == "little":
            headerdata = struct.unpack("<6i", f.read(24))
        else:
            headerdata = struct.unpack(">6i", f.read(24))

        headerdata_dict = {
            "order": headerdata[0],
            "firstyear": headerdata[1],
            "nyear": headerdata[2],
            "firstcell": headerdata[3],
            "ncell": headerdata[4],
            "nbands": headerdata[5],
        }

        if version == 2:
            if endian == "little":
                extra_data = struct.unpack("<2f", f.read(8))
            else:
                extra_data = struct.unpack(">2f", f.read(8))
            headerdata_dict.update(
                {"cellsize_lon": extra_data[0], "scalar": extra_data[1]}
            )

        if version >= 3:
            if endian == "little":
                extra_data = struct.unpack("<3f", f.read(12))
                datatype = struct.unpack("<i", f.read(4))[0]
            else:
                extra_data = struct.unpack(">3f", f.read(12))
                datatype = struct.unpack(">i", f.read(4))[0]
            headerdata_dict.update(
                {
                    "cellsize_lon": extra_data[0],
                    "scalar": extra_data[1],
                    "cellsize_lat": extra_data[2],
                    "datatype": datatype,
                }
            )

        if version == 4:
            if endian == "little":
                nstep = struct.unpack("<i", f.read(4))[0]
                timestep = struct.unpack("<i", f.read(4))[0]
            else:
                nstep = struct.unpack(">i", f.read(4))[0]
                timestep = struct.unpack(">i", f.read(4))[0]
            headerdata_dict.update({"nstep": nstep, "timestep": timestep})
        else:
            if len(headerdata_dict) == 6:
                headerdata_dict.update(
                    {
                        "cellsize_lon": 0.5,
                        "scalar": 1,
                        "cellsize_lat": 0.5,
                        "datatype": 1,
                        "nstep": 1,
                        "timestep": 1,
                    }
                )
                if verbose:
                    print(
                        "Note: Type 1 header. Adding default values for"
                        + " cellsize, scalar, datatype, nstep and timestep"
                        + " which may not be correct in all cases."
                    )
            if len(headerdata_dict) == 8:
                headerdata_dict.update(
                    {
                        "cellsize_lat": headerdata_dict["cellsize_lon"],
                        "datatype": 1,
                        "nstep": 1,
                        "timestep": 1,
                    }
                )
                if verbose:
                    print(
                        "Note: Type 2 header. Adding default values for"
                        + " datatype, nstep and timestep which may not be"
                        + " correct in all cases."
                    )
            if len(headerdata_dict) == 10:
                headerdata_dict.update({"nstep": 1, "timestep": 1})
                if verbose:
                    print(
                        "Note: Type 3 header. Adding default values for"
                        + " nstep and timestep which may not be correct in all"
                        + " cases."
                    )

        if verbose and headerdata_dict.get("datatype") is None:
            print(
                f"Warning: Invalid datatype {headerdata_dict['datatype']} in"
                + f" header read from {filename}"
            )

    if to_dict:
        return {
            "name": headername,
            "header": {"version": version, **headerdata_dict},
            "endian": endian,
        }
    else:
        return LPJmLMetaData(
            {
                "sim_name": None,
                "source": None,
                "history": None,
                "variable": headername,
                "long_name": None,
                "unit": None,
                "nbands": headerdata_dict["nbands"],
                "band_names": None,
                "nyear": headerdata_dict["nyear"],
                "firstyear": headerdata_dict["firstyear"],
                "lastyear": headerdata_dict["firstyear"]
                + headerdata_dict["nyear"]
                - 1,  # noqa
                "cellsize_lon": headerdata_dict["cellsize_lon"],
                "cellsize_lat": headerdata_dict.get(
                    "cellsize_lat", headerdata_dict["cellsize_lon"]
                ),  # noqa
                "ncell": headerdata_dict["ncell"],
                "firstcell": headerdata_dict["firstcell"],
                "nstep": headerdata_dict["nstep"],
                "timestep": headerdata_dict["timestep"],
                "datatype": headerdata_dict["datatype"],
                "scalar": headerdata_dict["scalar"],
            }
        )


# Function has been derived from the lpjmlkit R package
#   https://github.com/PIK-LPJmL/lpjmlkit
#   Author of original R function: Sebastian Ostberg
def get_headersize(filename):
    """
    Get the size of the header in an LPJmL input/output file.

    Parameters
    ----------
    filename : str
        Filename to read header size from.

    Returns
    -------
    int
        Size of the header in bytes.
    """
    header = read_header(filename, to_dict=True)
    version = header["header"]["version"]
    if version < 1 or version > 4:
        raise ValueError(
            "Invalid header version. Expecting value between 1 and 4."
        )  # noqa: E501

    headersize = len(header["name"]) + {1: 7, 2: 9, 3: 11, 4: 13}[version] * 4
    return headersize
