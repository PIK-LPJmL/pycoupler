import json
import numpy as np
import xarray as xr

from collections.abc import Hashable
from enum import Enum
from scipy.spatial import KDTree


class LPJmLInputType(Enum):
    """Available Input types"""
    landuse: int = 6  # number of bands in landuse data
    fertilizer_nr: int = 18  # number of bands in fertilizer data
    manure_nr: int = 19  # number of bands in manure data
    residue_on_field: int = 8  # number of bands in residue data
    with_tillage: int = 7  # number of bands in tillage data

    @property
    def nband(self):
        """ get amount of bands
        """
        if self.name == "landuse":
            return 64
        elif self.name in ["fertilizer_nr", "manure_nr", "residue_on_field"]:
            return 32
        else:  # "with_tillage"
            return 1

    @property
    def type(self):
        """ get amount of bands
        """
        if self.name in ["with_tillage"]:
            return int
        else:
            return float

    @property
    def bands(self):
        """ check if multiple bands - better check for categorical bands
        (ADJUST WHEN REQUIRED)
        """
        if self.nband > 1:
            return True
        else:
            return False


def append_to_dict(data_dict, data):
    """
    Append data along the third dimension to the data_dict.

    :param data_dict: Dictionary holding the data.
    :type data_dict: dict

    :param data: Dictionary with data.
                 Keys are ids/names, values are two-dimensional NumPy
                 arrays with dimensions (cells, bands).
    :type data: dict

    :return: Updated data_dict with the appended data.
    :rtype: dict
    """
    for key, value in data.items():
        if key in data_dict:
            data_dict[key] = np.dstack((data_dict[key], value))

        else:
            data_dict[key] = value

    return data_dict


class LPJmLData(xr.DataArray):
    """ Class for LPJmL data """
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLData, self).__init__(*args, **kwargs)

    def add_meta(self, meta_data):

        if isinstance(meta_data, LPJmLMetaData):

            self.attrs['standard_name'] = meta_data.variable
            self.attrs['long_name'] = meta_data.descr
            self.attrs['units'] = meta_data.unit
            self.attrs['source'] = meta_data.source
            self.attrs['history'] = meta_data.history

            if meta_data.cellsize_lat != meta_data.cellsize_lon:
                raise ValueError(
                    "Cell sizes in latitude and longitude direction must be "
                    "equal!"
                )
            else:
                self.attrs['cellsize'] = meta_data.cellsize_lon

            if hasattr(meta_data, 'global_attrs'):
                self.attrs['institution'] = meta_data.global_attrs.institution
                self.attrs['contact'] = meta_data.global_attrs.contact
                self.attrs['comment'] = meta_data.global_attrs.comment
        else:
            raise TypeError(
                "meta_data must be of type LPJmLMetaData!"
            )

    def get_neighbourhood(self, id=True, cellsize=0.5):
        """
        Get the IDs of all neighbouring cells within a given size of cells.
        :param id: If True, return cell ids, else return cell indices
        :type id: bool
        :param cellsize: Size of cells in degrees.
        :type cellsize: float
        :return: Array with the IDs of all neighbouring cells.
        :rtype: numpy.ndarray
        """

        # Get the coordinates of all cells
        coords_np = np.array([self.cell.latitude.values,
                              self.cell.longitude.values]).T

        # Build a KDTree for fast nearest-neighbour lookup
        tree = KDTree(coords_np)

        if "cellsize" in self.attrs:
            cellsize = self.cellsize  # in degrees

        # Find all neighbours within the given size of cells
        neighbour_indices = tree.query_ball_point(coords_np, r=cellsize)

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
                neighbour_ids[i, :len(current_neighbours)] = cell_indices[
                    current_neighbours
                ]
            elif len(current_neighbours) > 0 and not id:
                neighbour_ids[i, :len(current_neighbours)] = current_neighbours

        neighbours = LPJmLData(
            data=neighbour_ids,
            dims=('cell', 'neighbour'),
            coords=dict(
                cell=self.cell.values,
                longitude=(
                    ['cell'], self.longitude.values
                ),
                latitude=(
                    ['cell'], self.latitude.values
                ),
                neighbour=np.arange(8),
            ),
            name="neighbourhood"
        )

        return neighbours

    def transform(self):
        # TODO: implement function to convert cell into lon/lat format
        pass


class LPJmLDataSet(xr.Dataset):
    """Class for LPJmL data sets."""
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(LPJmLDataSet, self).__init__(*args, **kwargs)

        if self.data_vars and (
            "cellsize" in self[list(self.data_vars)[0]].attrs
        ):
            first_attrs = self[list(self.data_vars)[0]].attrs
            self.attrs['source'] = first_attrs["source"]
            self.attrs['history'] = first_attrs["history"]
            self.attrs['cellsize'] = first_attrs["cellsize"]

            if "institution" in first_attrs:
                self.attrs['institution'] = first_attrs["institution"]
                self.attrs['contact'] = first_attrs["contact"]
                self.attrs['comment'] = first_attrs["comment"]

    def to_numpy(self):
        """Return data as numpy array."""
        return {key: value.to_numpy() for key, value in self.data_vars.items()}

    def _construct_dataarray(self, name: Hashable) -> LPJmLData:
        """Construct a LPJmLData by indexing this dataset"""

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = xr.core.dataset._get_virtual_variable(
                self._variables, name, self.dims
            )

        needed_dims = set(variable.dims)

        coords: dict[Hashable, xr.core.variable.Variable] = {}
        # preserve ordering
        for k in self._variables:
            if k in self._coord_names and (
                set(self.variables[k].dims) <= needed_dims
            ):
                coords[k] = self.variables[k]

        indexes = xr.core.indexes.filter_indexes_from_coords(
            self._indexes, set(coords)
        )

        # TODO: this is a hack to get around the fact that we don't have
        #  a proper way to represent band dimensions in xarray

        # get the corresponding band dimension
        band_dim = [
            dim for dim in variable._dims
            if dim.startswith('band') and dim != 'band'
        ]
        if band_dim:
            variable._dims = variable._parse_dimensions([
                dim if not dim.startswith('band') else 'band'
                for dim in variable._dims
            ])

        # get the corresponding "band" index and delete all other band indexes
        band_idx = [
            key for key in coords.keys()
            if key.startswith("band") and key != "band"
        ]
        if band_idx:
            for key in band_idx:
                if name not in key:
                    del indexes[key]
                else:
                    band_idx_name = key
            indexes['band'] = indexes.pop(band_idx_name)
            indexes['band'].index.name = "band"

        # get the corresponding "band" coords and delete all other band coords
        band_coords = [
            key for key in coords
            if key.startswith("band") and key != "band"
        ]
        if band_coords:
            for key in band_coords:
                if name not in key:
                    del coords[key]
                else:
                    band_coords_name = key
            coords['band'] = coords.pop(band_coords_name)
            coords['band']._dims = ('band',)

        # rename the band dimension to only name "band"
        if name.startswith("band") and name != "band":
            name = "band"

        return LPJmLData(
            variable, coords, name=name, indexes=indexes, fastpath=True
        )


def read_data(file_name, var_name=None):
    """Read netcdf file and return data as numpy array or xarray.DataArray.
    :param file_name: path to netcdf file
    :type file_name: str
    :param var_name: name of variable to be read
    :type var_name: str
    :return: data as LPJmLData (xarray.DataArray)
    :rtype: LPJmLData
    """
    with xr.open_dataset(file_name,
                         decode_times=True,
                         mask_and_scale=False) as data:
        if var_name:
            data = data[var_name]
            data = LPJmLData(data)

        return data


class LPJmLMetaData:
    """LPJmL meta data class that can be easily accessed,
    converted to a dictionary or written as a json file.

    :param config_dict: takes a dictionary (ideally LPJmL config dictionary)
        and builds up a nested LpjmLConfig class with corresponding fields
    :type config_dict: dict
    """
    def __init__(self, meta_dict):
        """Constructor method
        """
        self.__dict__.update(meta_dict)
        if "band_names" not in self.__dict__:
            self.band_names = [str(ii) for ii in range(self.nbands)]

    def to_dict(self):
        """Convert class object to dictionary
        """
        def obj_to_dict(obj):
            if not hasattr(obj, '__dict__'):
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
        """Representation of the LPJmL Meta object
        """
        summary_attr = ["sim_name", "source", "history", "variable", "descr",
                        "unit", "nbands", "band_names", "nyear", "firstyear",
                        "lastyear", "cellsize_lon", "cellsize_lat", "ncell",
                        "firstcell", "datatype", "scalar"]
        other_attr = [
            to_repr for to_repr in self.__dict__.keys()
            if to_repr not in summary_attr
        ]
        if sub_repr:
            spacing = "\n  "
            summary = "Meta data:"
        else:
            summary = f"<pycoupler.{self.__class__.__name__}>"
            spacing = "\n"
        summary = spacing.join([
            summary,
            f"  * sim_name      {self.sim_name}",
            f"  * source        {self.source}",
            f"  * history       {self.history}",
            f"  * variable      {self.variable}",
            f"  * descr         {self.descr}",
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

        ])
        if other_attr:
            summary_list = [summary]
            summary_list.extend([
                f"  * {torepr}{(13-len(torepr))*' '} {getattr(self, torepr)}"
                for torepr in other_attr
            ])
            summary = spacing.join(summary_list)
        return summary


def read_meta(file_name, ):
    """Read meta data from json file and return as dictionary.
    :param file_name: path to json file
    :type file_name: str
    :return: meta data as dictionary
    """
    with open(file_name, 'r') as f:
        return json.load(f, object_hook=LPJmLMetaData)
