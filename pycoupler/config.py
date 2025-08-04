"""Classes and functions to handle LPJmL configurations and related operations"""

import os
import sys
import subprocess
import json
from subprocess import run
from ruamel.yaml import YAML

from pycoupler.utils import read_json, get_countries, create_subdirs, detect_io_type
from pycoupler.data import read_header


class SubConfig:
    """
    LPJmL sub config class for easy access, conversion to a dictionary,
    or writing as a JSON file.

    Parameters
    ----------
    config_dict : dict
        Dictionary (ideally an LPJmL config dictionary) used to build up a nested
        LpjmLConfig class with corresponding fields.
    """

    def __init__(self, config_dict):
        """Constructor method"""
        self.__dict__.update(config_dict)

    def to_dict(self):
        """
        Convert class object to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the class object.
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

    def __iter__(self):
        """
        Iterate over items of a SubConfig.

        Yields
        ------
        tuple
            Key-value pairs of the SubConfig attributes (excluding private
            attributes).
        """
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                yield key, value

    def to_json(self, file_name=None):
        """
        Write the configuration to a JSON file.

        Parameters
        ----------
        file_name : str, optional
            File name (including relative/absolute path) to write JSON to.
            If None, a default name is used.

        Returns
        -------
        str
            File name of the written JSON file.
        """
        # convert class to dict
        config_dict = self.to_dict()

        # configuration file name
        if file_name is None:
            json_file = f"{self.sim_path}/config_{self.sim_name}.json"
        else:
            json_file = file_name

        # write json and prettify via indent
        with open(json_file, "w") as con:
            json.dump(config_dict, con, indent=2)

        return json_file


class LpjmlConfig(SubConfig):
    """
    LPJmL config class for easy access, conversion to a dictionary,
    or writing as a JSON file. Provides methods to get/set outputs,
    restarts, and sockets for model coupling.

    Parameters
    ----------
    sub_config : SubConfig
        SubConfig instance (ideally built from an LPJmL config dictionary)
        used to build up a nested LpjmlConfig class with corresponding fields.
    """

    def __init__(self, sub_config):
        """Constructor method"""
        # add changed attribute to sub config to track config changes
        if "changed" not in sub_config.__dict__:
            sub_config.__dict__["changed"] = []
        self.__dict__.update(sub_config.__dict__)

    def get_output_avail(self, id_only=True, to_dict=False):
        """
        Get available output (outputvar) names or objects.

        Parameters
        ----------
        id_only : bool, default True
            If True, only output IDs are returned. If False, the whole
            outputvar objects are returned.
        to_dict : bool, default False
            If True, a dictionary is returned. If False, a list with the
            config objects of the outputvar.

        Returns
        -------
        list or dict
            List of output IDs (if id_only=True and to_dict=False),
            dict of output objects (if to_dict=True),
            or list of outputvar objects (if id_only=False and to_dict=False).
        """
        if id_only:
            return [out.name for out in self.outputvar]
        else:
            if to_dict:
                return {out.name: out.to_dict() for out in self.outputvar}
            else:
                return self.outputvar

    def get_output(self, id_only=True, to_dict=False, fmt=None):
        """
        Get defined output IDs or objects.

        Parameters
        ----------
        id_only : bool, default True
            If True, only output IDs are returned. If False, the whole
            output objects are returned.
        to_dict : bool, default False
            If True, a dictionary is returned. If False, a list with the
            config objects of the output.
        fmt : str, optional
            If specified, only outputs with the defined file format are
            returned.

        Returns
        -------
        list or dict
            List of output IDs (if id_only=True and to_dict=False),
            dict of output objects (if to_dict=True),
            or list of output objects (if id_only=False and to_dict=False).
        """
        if fmt:
            if id_only:
                return [out.id for out in self.output if (out.file.fmt == fmt)]
            else:
                outs = [
                    out
                    for pos, out in enumerate(self.output)
                    if out.file.fmt == fmt  # noqa
                ]
        else:
            if id_only:
                return [out.id for out in self.output]
            else:
                outs = self.output

        if to_dict:
            return {out.id: out.to_dict() for out in outs}
        else:
            return outs

    def set_spinup(self, sim_path, sim_name="spinup"):
        """
        Set configuration required for spinup model runs.

        Parameters
        ----------
        sim_path : str
            Path where simulation data is written to.
        sim_name : str, default "spinup"
            Name of the simulation.
        """
        self.sim_name = sim_name
        self.sim_path = create_subdirs(sim_path, self.sim_name)
        output_path = f"{sim_path}/output/{self.sim_name}"

        # set output writing
        self._set_outputpath(output_path)
        # set restart directory to restart from in subsequent historic run
        self._set_restart(path=f"{sim_path}/restart")

    def set_transient(
        self,
        sim_path,
        start_year,
        end_year,
        sim_name="transient",
        dependency=None,
        temporal_resolution="annual",
        write_output=[],
        write_file_format="cdf",
        append_output=True,
    ):
        """
        Set configuration required for historic model runs.

        Parameters
        ----------
        sim_path : str
            Path where simulation data is written to.
        start_year : int
            Start year of simulation.
        end_year : int
            End year of simulation.
        sim_name : str, default "transient"
            Name of the simulation.
        dependency : str, optional
            Name of simulation to depend on (e.g., spinup run).
        temporal_resolution : str or dict, default "annual"
            Temporal resolution for outputs. Can be a dict of temporal resolutions
            corresponding to outputs or a str to set the same resolution for
            all outputs. Choose between "annual", "monthly", "daily".
        write_output : list, default []
            Output IDs to be written by LPJmL. Check available outputs with
            get_output_avail().
        write_file_format : str, default "cdf"
            File format of output files. Choose between "raw", "clm", and "cdf".
        append_output : bool, default True
            If True, defined output entries are appended. If False, existing
            outputs are overwritten.
        """
        self.sim_name = sim_name
        self.sim_path = create_subdirs(sim_path, self.sim_name)
        output_path = f"{sim_path}/output/{self.sim_name}"
        # set time range for historic run
        self._set_timerange(
            start_year=start_year, end_year=end_year, write_start_year=start_year
        )
        # set output writing
        self._set_output(
            output_path,
            outputs=write_output,
            temporal_resolution=temporal_resolution,
            file_format=write_file_format,
            append_output=append_output,
        )
        # set start from directory to start from spinup run
        self._set_startfrom(path=f"{sim_path}/restart", dependency=dependency)
        # set restart directory to restart from in subsequent transient run
        self._set_restart(path=f"{sim_path}/restart")

    def set_coupled(
        self,
        sim_path,
        start_year,
        end_year,
        coupled_input,
        coupled_output,
        sim_name="coupled",
        dependency=None,
        coupled_year=None,
        temporal_resolution="annual",
        write_output=[],
        write_file_format="cdf",
        append_output=True,
        model_name="copan:CORE",
    ):
        """
        Set configuration required for coupled model runs.

        Parameters
        ----------
        sim_path : str
            Path where simulation data is written to.
        start_year : int
            Start year of simulation.
        end_year : int
            End year of simulation.
        coupled_input : list
            List of inputs to be used as socket for coupling.
            Provide dictionary/json key as identifier.
        coupled_output : list
            List of outputs to be used as socket for coupling.
            Provide output ID as identifier.
        sim_name : str, default "coupled"
            Name of the simulation.
        dependency : str, optional
            Name of simulation to depend on (e.g., transient run).
        coupled_year : int, optional
            Start year of coupled simulation. If None, uses start_year.
        temporal_resolution : str or dict, default "annual"
            Temporal resolution for outputs. Can be a dict of temporal
            resolutions corresponding to outputs or a str to set the same
            resolution for all outputs. Choose between "annual", "monthly",
            "daily".
        write_output : list, default []
            Output IDs to be written by LPJmL. Check available outputs with
            get_output_avail().
        write_file_format : str, default "cdf"
            File format of output files. Choose between "raw", "clm", and "cdf".
        append_output : bool, default True
            If True, defined output entries are appended. If False, existing
            outputs are overwritten.
        model_name : str, default "copan:CORE"
            Name of the coupled model.
        """
        self.sim_name = sim_name
        self.sim_path = create_subdirs(sim_path, self.sim_name)
        output_path = f"{sim_path}/output/{self.sim_name}"

        # set time range for coupled run
        self._set_timerange(
            start_year=start_year, end_year=end_year, write_start_year=start_year
        )
        # set grid explicitly to be able to use start and endgrid
        self._set_grid_explicitly()

        # set output directory, outputs (relevant ones for pbs and agriculture)
        write_output += [
            item for item in coupled_output if item not in write_output
        ]  # noqa
        self._set_output(
            output_path,
            outputs=write_output,
            temporal_resolution=temporal_resolution,
            file_format=write_file_format,
            append_output=append_output,
        )
        # set coupling parameters
        self._set_coupling(
            inputs=coupled_input,
            outputs=coupled_output,
            start_year=coupled_year,
            model_name=model_name,
        )
        # set start from directory to start from historic run
        self._set_startfrom(path=f"{sim_path}/restart", dependency=dependency)

    def _set_output(
        self,
        output_path,
        outputs=[],
        file_format="raw",
        temporal_resolution="annual",
        append_output=True,
    ):
        """
        Set outputs to be written by LPJmL and define temporal resolution.

        Parameters
        ----------
        output_path : str
            Path where output files are written to. If append_output=True,
            output_path is only altered for appended outputs.
        outputs : list, default []
            Output IDs to be written by LPJmL. Check available outputs with
            get_output_avail().
        file_format : str, default "raw"
            File format for outputs (not for sockets). Available formats:
            "raw" (binary), "clm" (binary with header), "cdf" (NetCDF).
        temporal_resolution : str or dict, default "annual"
            Temporal resolution for outputs. Can be a dict of resolutions
            corresponding to outputs or a str to set the same resolution
            for all outputs. Choose from "annual", "monthly", "daily".
        append_output : bool, default True
            If True, defined output entries are appended. If False,
            existing outputs are overwritten.
        """
        available_res = ("annual", "monthly", "daily")
        available_formats = {"raw": "bin", "clm": "clm", "cdf": "nc4"}
        nonvariable_outputs = "globalflux"

        # provide additional meta data
        self.output_metafile = True
        self.grid_type = "float"

        # add grid output if not already defined
        if "grid" not in outputs:
            outputs.append("grid")

        # create dict of outputvar names with indexes for iteration
        outputvar_names = {ov.name: pos for pos, ov in enumerate(self.outputvar)}
        # extract dict of outputvar for manipulation
        outputvars = self.to_dict()["outputvar"]

        if not append_output:
            self.output = list()
        else:
            # create dict of output names with indexes for iteration
            output_names = list()
            # modify existing output entries
            for pos, out in enumerate(self.output):

                output_names.append(out.id)
                # self.output[pos].file.socket = False
                # Only change temporal_resolution if string, so not
                # specifically for each output
                if isinstance(temporal_resolution, str):
                    self.output[pos].file.timestep = temporal_resolution
                elif (
                    isinstance(temporal_resolution, dict)
                    and out.id in temporal_resolution.keys()
                ):
                    self.output[pos].file.timestep = temporal_resolution[out.id]

                if out.id not in nonvariable_outputs:
                    self.output[pos].file.fmt = file_format
                    self.output[pos].file.name = (
                        f"{output_path}/{out.id}"
                        f".{available_formats[file_format]}"  # noqa
                    )
                else:
                    self.output[pos].file.name = (
                        f"{output_path}/{out.id}"
                        f".{os.path.splitext(self.output[pos].file.name)[1]}"
                    )

        # handle each defined output
        for pos, out in enumerate(outputs):
            if out in outputvar_names and out not in output_names:

                # check if temporal resolution is defined for output
                if isinstance(temporal_resolution, str):
                    timestep = temporal_resolution
                elif (
                    isinstance(temporal_resolution, dict)
                    and out in temporal_resolution.keys()
                ):
                    timestep = temporal_resolution[out]
                else:
                    timestep = outputvars[outputvar_names[out]]["timestep"]

                # create new output entry
                new_out = SubConfig(
                    {
                        "id": outputvars[outputvar_names[out]]["name"],
                        "file": SubConfig(
                            {
                                "fmt": file_format,
                                # 'socket': False,
                                "timestep": timestep,
                                "name": f"{output_path}/"
                                f"{outputvars[outputvar_names[out]]['name']}"
                                f".{available_formats[file_format]}",
                            }
                        ),
                    }
                )
                self.output.append(new_out)
            elif out not in outputvar_names:
                # raise error if defined outputs are not in outputvar
                raise ValueError(
                    f"The following output is not defined in outputvar: {out}"
                )

    def _set_outputpath(self, output_path):
        """Set output path of specified outputs"""
        for out in self.output:
            file_name = out.file.name.split("/")
            file_name.reverse()
            out.file.name = f"{output_path}/{file_name[0]}"

    def _set_startfrom(self, path, dependency=None):
        """Set restart file from which LPJmL starts the transient run"""
        if dependency:
            self.restart_filename = f"{path}/restart_{dependency}.lpj"

        else:
            file_name = self.restart_filename.split("/")
            file_name.reverse()
            self.restart_filename = f"{path}/{file_name[0]}"

    def _set_restart(self, path):
        """Set restart file from which LPJmL starts the transient run"""
        self.write_restart_filename = f"{path}/restart_{self.sim_name}.lpj"
        self.restart_year = self.lastyear

    def _set_timerange(
        self, start_year=1901, end_year=2017, write_start_year=None
    ):  # noqa
        """Set simulation time range, outputyear to start as a default here."""
        self.firstyear = start_year
        self.lastyear = end_year
        if write_start_year:
            self.outputyear = write_start_year
        else:
            self.outputyear = start_year

    def _set_grid_explicitly(self, only_all=True):
        """Set startgrid and endgrid for LPJmL simulation"""
        if self.startgrid == "all":
            self.startgrid = 0
        if self.endgrid == "all" or not only_all:
            if hasattr(sys, "_called_from_test"):
                self.endgrid = 2
            elif self.input.soil.fmt in ["json", "meta"]:
                self.endgrid = (
                    read_json(
                        (
                            self.input.soil.name
                            if os.path.isfile(self.input.soil.name)
                            else f"{self.inpath}/{self.input.soil.name}"
                        )
                    )["ncell"]
                    - 1
                )
            else:
                self.endgrid = (
                    read_header(
                        (
                            self.input.soil.name
                            if os.path.isfile(self.input.soil.name)
                            else f"{self.inpath}/{self.input.soil.name}"
                        ),
                        to_dict=True,
                    )["header"]["ncell"]
                    - 1
                )

    def _set_coupling(
        self, inputs, outputs, start_year=None, model_name="copan:CORE"
    ):  # noqa
        """Coupled settings - no spinup, not write restart file and set sockets"""
        self.write_restart = False
        self.nspinup = 0
        self.float_grid = True
        self.coupled_model = model_name
        self._set_input_sockets(inputs)
        self._set_outputsockets(outputs)
        if start_year:
            self.start_coupling = start_year
        else:
            self.start_coupling = self.firstyear

    def _set_input_sockets(self, inputs=[]):
        """Set sockets for inputs and outputs (via corresponding ids)"""
        for inp in inputs:
            sock_input = getattr(self.input, inp)
            if "id" not in sock_input.__dict__.keys():
                raise ValueError("Please use a config with input ids.")
            sock_input.__dict__["socket"] = True

    def _set_outputsockets(self, outputs=[]):
        """Set sockets for inputs and outputs (via corresponding ids)"""
        if "grid" not in outputs:
            outputs.append("grid")

        # get names/ids only of outputs that are defined in outputvar
        valid_outs = {out.name for out in self.outputvar if out.name in outputs}

        # check if all outputs are valid
        nonvalid_outputs = list(set(outputs) - valid_outs)
        if nonvalid_outputs:
            raise ValueError(
                "The following outputs are not defined in outputvar: "
                f"{nonvalid_outputs}"
            )
        # get position of valid outputs in config output list
        output_pos = [
            pos for pos, out in enumerate(self.output) if out.id in valid_outs
        ]

        # set socket to true for corresponding outputs
        for pos in output_pos:
            if self.output[pos].id in valid_outs:
                self.output[pos].file.socket = True

    def get_input_sockets(self, id_only=False):
        """Get defined socket inputs as dict

        Parameters
        ----------
        id_only : bool, default False
            If True, only input IDs are returned. If False, all input
            information is returned.
        """
        inputs = self.input.to_dict()
        if id_only:
            return [
                inp
                for inp in inputs
                if ("socket" in inputs[inp]) and inputs[inp]["socket"]
            ]
        else:
            return {
                inp: inputs[inp]
                for inp in inputs
                if ("socket" in inputs[inp]) and inputs[inp]["socket"]
            }

    def get_output_sockets(self, id_only=False):
        """Get defined socket outputs as dict

        Parameters
        ----------
        id_only : bool, default False
            If True, only output IDs are returned. If False, all output
            information is returned.
        """
        outputs = self.to_dict()["output"]
        name_id = {out.name: out.id for out in self.outputvar}

        if id_only:
            return [
                out["id"]
                for out in outputs
                if ("socket" in out["file"]) and out["file"]["socket"]
            ]
        else:
            return {
                out["id"]: dict({"index": name_id[out["id"]]}, **out)
                for out in outputs
                if ("socket" in out["file"]) and out["file"]["socket"]
            }

    def add_config(self, file_name):
        """Add config file of coupled model to LPJmL config

        Parameters
        ----------
        file_name : str
            Path to coupled config file
        """
        self.coupled_config = read_yaml(file_name, CoupledConfig)

    def regrid(self, sim_path, model_path=None, country_code="BEL", overwrite=False):
        """
        Regrid LPJmL configuration file to a new country.

        Parameters
        ----------
        sim_path : str
            Directory to check whether required subfolders exist. If not,
            create corresponding folders (input, output, restart).
        model_path : str, optional
            Path to LPJmL_internal (lpjml repository). If None, uses
            self.model_path.
        country_code : str, default "BEL"
            Country code of country to regrid to.
        overwrite : bool, default False
            Whether to overwrite existing country-specific input files.

        Raises
        ------
        OSError
            If sim_path or model_path do not exist, or if required grid files
            are missing.
        """

        if not os.path.exists(sim_path):
            raise OSError(f"Path '{sim_path}' does not exist.")

        if hasattr(self, "model_path"):
            model_path = self.model_path
        elif not model_path or not os.path.exists(model_path):
            raise OSError(f"Path '{model_path}' does not exist.")

        # get available countries of LPJmL
        countries = get_countries()

        # get country name from country code
        country = next(
            (
                countries[country]["name"]
                for country in countries
                if (countries[country]["code"] == country_code)
            ),
            None,
        ).lower()

        grid_file = (
            self.input.coord.name
            if os.path.isfile(self.input.coord.name)
            or hasattr(sys, "_called_from_test")
            else f"{self.inpath}/{self.input.coord.name}"
        )

        # proxy check if regrid was already performed
        if country in self.input.coord.name:
            return

        country_grid_file = (
            f"{sim_path}/input/{country}_{os.path.basename(self.input.coord.name)}"
        )
        # check if country specific input files already exist
        if (not os.path.isfile(country_grid_file) or overwrite) and not hasattr(
            sys, "_called_from_test"
        ):

            if not os.path.isfile(grid_file):
                raise OSError(f"Grid file '{grid_file}' does not exist.")

            # extract country specific grid
            run(
                [
                    f"{model_path}/bin/getcountry",
                    (
                        self.input.countrycode.name
                        if os.path.isfile(self.input.countrycode.name)
                        else f"{self.inpath}/{self.input.countrycode.name}"
                    ),
                    grid_file,
                    country_grid_file,
                    country_code,
                ],
                check=True,
            )

        self.input.coord.fmt = (
            detect_io_type(country_grid_file)
            if not hasattr(sys, "_called_from_test")
            else "clm"
        )
        self.input.coord.name = country_grid_file

        lakes_fn_string = (
            self.input.lakes.name
            if os.path.isfile(self.input.lakes.name)
            or hasattr(sys, "_called_from_test")
            else f"{self.inpath}/{self.input.lakes.name}"
        )
        # extract country specific lakes file from meta file
        if self.input.lakes.fmt == "meta" and not hasattr(sys, "_called_from_test"):
            lakes_filename = read_json(lakes_fn_string)["filename"]

            lakes_file = lakes_fn_string
            lakes_file = (
                f"{lakes_file[:lakes_file.rfind('/')+1]}{lakes_filename}"  # noqa
            )
        else:
            lakes_file = lakes_fn_string

        country_lakes_file = (
            f"{sim_path}/input/{country}_{os.path.basename(lakes_file)}"
        )

        # check if country specific input files already exist
        if (not os.path.isfile(country_lakes_file) or overwrite) and not hasattr(
            sys, "_called_from_test"
        ):

            if not os.path.isfile(lakes_file):
                raise OSError(f"Lakes file '{lakes_file}' does not exist.")

            # regrid lakes file to country specific grid
            run(
                [
                    f"{model_path}/bin/regridsoil",
                    grid_file,
                    country_grid_file,
                    lakes_file,
                    country_lakes_file,
                ],
                check=True,
                stdout=open(os.devnull, "wb"),
            )

        self.input.lakes.fmt = (
            detect_io_type(country_lakes_file)
            if not hasattr(sys, "_called_from_test")
            else "raw"
        )
        self.input.lakes.name = country_lakes_file

        # loop over all used input files to regrid them to country specific
        #   grid
        for config_key, config_input in self.input:

            if (
                config_input.fmt != "clm"
                or config_key in ["coord", "lakes"]
                or (config_input.name == "DUMMYLOCATION")
            ):
                continue

            input_file = (
                config_input.name
                if os.path.isfile(config_input.name)
                or hasattr(sys, "_called_from_test")
                else f"{self.inpath}/{config_input.name}"
            )

            country_input_file = (
                f"{sim_path}/input/{country}_{os.path.basename(input_file)}"
            )

            # check if country specific input files already exist
            if (not os.path.isfile(country_input_file) or overwrite) and not hasattr(
                sys, "_called_from_test"
            ):

                if not os.path.isfile(input_file):
                    raise OSError(f"Input file '{input_file}' does not exist.")

                if config_key == "drainage":
                    regrid_func = "regriddrain"
                elif config_key == "irrigation":
                    regrid_func = "regridirrig"
                else:
                    regrid_func = "regridclm"

                # regrid all other input files to country specific grid
                regrid_cmd = [
                    f"{model_path}/bin/{regrid_func}",
                    grid_file,
                    self.input.coord.name,
                    input_file,
                    country_input_file,
                ]
                # if additional_arg:
                #     regrid_cmd.insert(1, additional_arg)
                run(regrid_cmd, check=True, stdout=open(os.devnull, "wb"))

            config_input.fmt = (
                detect_io_type(country_input_file)
                if not hasattr(sys, "_called_from_test")
                else "clm"
            )
            config_input.name = country_input_file

        self._set_grid_explicitly(only_all=False)

    def convert_cdf_to_raw(self, output_id=None):
        """Convert netcdf files to raw (binary) files.
        :param output_id: list with ids of outputs to convert from netcdf to
            raw
        :type output_id: list
        """

        output_dir = f"{self.sim_path}/output/{self.sim_name}"

        grid_file = (
            self.input.coord.name
            if os.path.isfile(self.input.coord.name)
            or hasattr(sys, "_called_from_test")
            else f"{self.inpath}/{self.input.coord.name}"
        )

        grid_name = os.path.basename(grid_file)

        if not os.path.isfile(f"{output_dir}/{grid_name}") and not hasattr(
            sys, "_called_from_test"
        ):
            run(f"tail -c +44 {grid_file} > {output_dir}/{grid_name}", shell=True)

        grid_file = f"{output_dir}/{grid_name}"

        outputs = [
            out for out in self.get_output(fmt="cdf", id_only=True) if out != "grid"
        ]

        if output_id:
            if isinstance(output_id, str):
                output_id = [output_id]  # Convert to list for consistency
            outputs = [out for out in outputs if out in output_id]

        output_details = [
            out
            for out in self.get_output_avail(id_only=False)
            if out.name in outputs  # noqa
        ]

        for output in output_details:
            # convert netcdf output to netcdf files
            conversion_cmd = [
                f"{self.model_path}/bin/cdf2bin",
                # "-units", output.unit,
                "-var",
                output.var,
                "-o",
                f"{output_dir}/{output.name}.bin",
                "-json",
                grid_file,
                f"{output_dir}/{output.name}.nc4",
            ]

            if None in conversion_cmd:
                conversion_cmd.remove(None)

            if not hasattr(sys, "_called_from_test"):
                run(conversion_cmd)

            nc4_meta_dict = read_json(f"{output_dir}/{output.name}.nc4.json")

            bin_meta_dict = read_json(f"{output_dir}/{output.name}.bin.json")

            for key, value in nc4_meta_dict.items():
                if key not in bin_meta_dict or key == "band_names":
                    bin_meta_dict[key] = value
                if key == "ref_area":
                    bin_meta_dict[key]["filename"] = (
                        nc4_meta_dict["ref_area"]["filename"].split(".")[0]
                        + ".bin.json"
                    )

            if hasattr(sys, "_called_from_test"):
                return "tested"

            with open(f"{output_dir}/{output.name}.bin.json", "w") as f:
                json.dump(bin_meta_dict, f, indent=2)

    def __repr__(self, sub_repr=0):
        """Representation of the config object"""
        summary_attr = [
            "sim_id",
            "sim_name",
            "version",
            "firstyear",
            "lastyear",
            "startgrid",
            "endgrid",
            "landuse",
            "coupled_model",
            "start_coupling",
            "coupled_config",
        ]
        changed_repr = [
            to_repr for to_repr in self.changed if to_repr not in summary_attr
        ]
        if sub_repr > 0:
            spacing = "\n" + "  " * sub_repr
            summary = "Configuration:"
        else:
            summary = f"<pycoupler.{self.__class__.__name__}>"
            spacing = "\n"
        summary = spacing.join(
            [
                summary,
                f"Settings:      {self.sim_id} v{self.version}",
                "  (general)",
                f"  * sim_name   {self.sim_name}",
                f"  * firstyear  {self.firstyear}",
                f"  * lastyear   {self.lastyear}",
                f"  * startgrid  {self.startgrid}",
                f"  * endgrid    {self.endgrid}",
                f"  * landuse    {self.landuse}",
            ]
        )
        if changed_repr:
            summary_list = [summary, "  (changed)"]
            summary_list.extend(
                [
                    f"  * {torepr}{(20-len(torepr))*' '} {getattr(self, torepr)}"
                    for torepr in changed_repr
                ]
            )
            summary = spacing.join(summary_list)

        if self.coupled_model:
            sc = self.start_coupling if self.start_coupling else self.firstyear
            input_coupled = self.get_input_sockets(id_only=True)
            output_coupled = self.get_output_sockets(id_only=True)

            if hasattr(self, "coupled_config") and isinstance(
                self.coupled_config, CoupledConfig
            ):
                coupled_config_repr = f"""  * coupled_config:  {
                    self.coupled_config.__repr__(sub_repr + 2)
                }"""
            else:
                coupled_config_repr = ""
            summary = spacing.join(
                [
                    summary,
                    f"Coupled model:        {self.coupled_model}",
                    f"  * start_coupling    {sc}",
                    f"  * input (coupled)   {input_coupled}",
                    f"  * output (coupled)  {output_coupled}",
                    coupled_config_repr,
                ]
            )

        return summary

    def __setattr__(self, __name, __value):
        super().__setattr__(__name, __value)
        self.changed.append(__name)


def parse_config(
    file_name="./lpjml_config.json", spin_up=False, macros=None, config_class=None
):
    """
    Precompile lpjml_config.json and return LpjmlConfig object or dict.

    Evaluates macros using the C preprocessor, analogous to R function
    lpjmlKit::parse_config.

    Parameters
    ----------
    file_name : str, default "./lpjml_config.json"
        Path to the LPJmL configuration file.
    spin_up : bool, default False
        Convenience argument to set macro whether to start from restart file
        (True) or not (False).
    macros : str or list, optional
        Macro(s) to provide in the form of "-DMACRO" or list of macros.
    config_class : class, optional
        Class of config object to be returned. If None, returns dict.

    Returns
    -------
    LpjmlConfig or dict
        A LpjmlConfig object if config_class is provided, otherwise a
        dictionary.
    """
    # precompile command
    cmd = ["cpp", "-P"]
    # add arguments
    if not spin_up:
        cmd.append("-DFROM_RESTART")
    if macros:
        if isinstance(macros, list):
            cmd.extend(macros)
        else:
            cmd.append(macros)
    cmd.append(file_name)

    # Subprocess call of cmd - return stdout
    json_str = subprocess.run(cmd, capture_output=True)

    # Convert to dict
    lpjml_config = json.loads(json_str.stdout, object_hook=config_class)

    return lpjml_config


def read_config(
    file_name, model_path=None, spin_up=False, macros=None, to_dict=False
):  # noqa
    """
    Read LPJmL configuration files and return as LpjmlConfig object or dict.

    Parameters
    ----------
    file_name : str
        File name (including relative/absolute path) of the LPJmL configuration.
    model_path : str, optional
        Path to model root directory. If provided, joined with file_name.
    spin_up : bool, default False
        Convenience argument to set macro whether to start from restart file
        (True) or not (False).
    macros : str or list, optional
        Macro(s) to provide in the form of "-DMACRO" or list of macros.
    to_dict : bool, default False
        If True, a dictionary is returned. If False, an LpjmlConfig object is
        returned.

    Returns
    -------
    LpjmlConfig or dict
        LpjmlConfig object (if to_dict=False) or dictionary (if to_dict=True).
    """
    if model_path is not None:
        file_name = os.path.join(model_path, file_name)

    if not to_dict:
        config = SubConfig
    else:
        config = None

    # Try to read file as json
    try:
        lpjml_config = read_json(file_name, object_hook=config)

    # If not possible, precompile and parse JSON
    except json.decoder.JSONDecodeError:
        lpjml_config = parse_config(
            file_name, spin_up=spin_up, macros=macros, config_class=config
        )

    # Convert first level to LpjmlConfig object
    if not to_dict:
        if hasattr(lpjml_config, "coupled_config") and isinstance(
            lpjml_config.coupled_config, SubConfig
        ):
            lpjml_config.coupled_config = CoupledConfig(
                lpjml_config.coupled_config.to_dict()
            )

            def convert_to_coupled_config(config):
                """Recursively converts nested dictionaries to CoupledConfig
                objects."""
                for key, value in config.items():
                    if isinstance(value, dict):
                        config[key] = CoupledConfig(value)
                        convert_to_coupled_config(config[key].__dict__)
                return config

            convert_to_coupled_config(lpjml_config.coupled_config.__dict__)

        lpjml_config = LpjmlConfig(lpjml_config)

    if model_path is not None:
        if to_dict:
            lpjml_config["model_path"] = model_path
        else:
            lpjml_config.model_path = model_path

    return lpjml_config


class CoupledConfig(SubConfig):
    """Class to handle coupled model configurations."""

    def __repr__(self, sub_repr=1, order=1):
        """Representation of the config object"""
        spacing = "\n" + "  " * sub_repr
        if order == 1 and sub_repr == 1:
            summary = f"Coupled configuration:{spacing}"
        else:
            summary = spacing

        for key, value in self.__dict__.items():
            if isinstance(value, SubConfig):
                summary += (
                    f"""{'  ' * sub_repr}* {key}: {value.__repr__(
                        sub_repr + 1, order + 1
                    )}""".strip()
                    + spacing
                )
            else:
                summary += (
                    f"{'  ' * sub_repr}* {key:<20} {value}".strip() + spacing
                )  # noqa

        return summary


def read_yaml(file_name, config_class):
    """Read YAML file and return as LpjmlConfig object or dict.

    Parameters
    ----------
    file_name : str
        Path to YAML file.
    config_class : class
        Class of config object to be returned. If None, returns dict.

    Returns
    -------
    LpjmlConfig or dict
        LpjmlConfig object (if config_class is provided) or dictionary.
    """
    with open(file_name, "r") as f:
        yaml = YAML(typ="safe", pure=True)
        yaml_data = yaml.load(f)

    return _from_yaml(yaml_data, config_class)


def _from_yaml(yaml_data, config_class):
    if isinstance(yaml_data, dict):
        return config_class(
            {k: _from_yaml(v, config_class) for k, v in yaml_data.items()}
        )
    elif isinstance(yaml_data, list):
        return [_from_yaml(v, config_class) for v in yaml_data]
    else:
        return yaml_data
