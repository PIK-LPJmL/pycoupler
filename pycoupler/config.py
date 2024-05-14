import os
import subprocess
import json
import ruamel.yaml
from subprocess import run

from pycoupler.utils import read_json, get_countries, create_subdirs


class SubConfig:
    """This serves as an LPJmL sub config class that can be easily accessed,
    converted to a dictionary or written as a json file.

    :param config_dict: takes a dictionary (ideally LPJmL config dictionary)
        and builds up a nested LpjmLConfig class with corresponding fields
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        """Constructor method
        """
        self.__dict__.update(config_dict)

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

    def __iter__(self):
        """Iteration method to get items of a SubConfig
        """
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                yield key, value

    def to_json(self, file_name=None):
        """Write json file
        :param file: file name (including relative/absolute path) to write json
            to
        :type: str
        :param file_name: file name (including relative/absolute path) to write
            json to
        :type file_name: str
        :return: file name of written json file
        :rtype: str
        """
        # convert class to dict
        config_dict = self.to_dict()

        # configuration file name
        if file_name is None:
            json_file = f"{self.sim_path}/config_{self.sim_name}.json"
        else:
            json_file = file_name

        # write json and prettify via indent
        with open(json_file, 'w') as con:
            json.dump(config_dict, con, indent=2)

        return json_file


class LpjmlConfig(SubConfig):
    """This serves as an LPJmL config class that can be easily accessed,
    converted to a dictionary or written as a json file. It further provides
    methods to get/set outputs, restarts and sockets for model coupling.

    :param config_dict: takes a dictionary (ideally LPJmL config dictionary)
        and builds up a nested LpjmLConfig class with corresponding fields
    :type config_dict: dict
    """

    def __init__(self, sub_config):
        """Constructor method
        """
        # add changed attribute to sub config to track config changes
        if "changed" not in sub_config.__dict__:
            sub_config.__dict__["changed"] = []
        self.__dict__.update(sub_config.__dict__)

    def get_output_avail(self, id_only=True, to_dict=False):
        """Get available output (outputvar) names (== output ids) as list

        :param id_only: if True only output ids are returned, else the whole
            outputvar object
        :type id_only: bool
        :param to_dict: if True a dictionary is returned, else a list with the
            config objects of the outputvar
        :type to_dict: bool
        """
        if id_only:
                return [out.name for out in self.outputvar]
        else:
            if to_dict:
                return {out.name: out.to_dict() for out in self.outputvar}
            else:    
                return self.outputvar

    def get_output(self, id_only=True, to_dict=False, fmt=None):
        """Get defined output ids as list

        :param id_only: if True only output ids are returned, else the whole
            output object
        :type id_only: bool
        :param to_dict: if True a dictionary is returned, else a list with the
            config objects of the output
        :type to_dict: bool
        :param fmt: if defined only outputs with defined file format are
            returned
        :type fmt: str
        """
        if fmt:
            if id_only:
                return [out.id for out in self.output if (
                    out.file.fmt == fmt
                )]
            else:
                outs = [
                    out for pos, out in enumerate(
                        self.output
                    ) if out.file.fmt == fmt
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

    def set_spinup(self, sim_path):
        """Set configuration required for spinup model runs
        :param sim_path: define sim_path data is written to
        :type sim_path: str
        :param restart_path: define restart_path the restart files to start
            model runs from. Has to match with `restart_path` from
            `set_transient`
        :type restart_path: str
        """
        self.sim_name = "spinup"
        self.sim_path = create_subdirs(sim_path, self.sim_name)
        output_path = f"{sim_path}/output/{self.sim_name}"

        # set output writing
        self.set_outputpath(output_path)
        # set restart directory to restart from in subsequent historic run
        self._set_restart(path=f"{sim_path}/restart")

    def set_transient(self,
                      sim_path,
                      start_year, end_year,
                      sim_name="transient",
                      dependency=None,
                      write_output=[],
                      write_temporal_resolution="annual",
                      write_file_format="cdf",
                      append_output=True):
        """Set configuration required for historic model runs
        :param sim_path: define sim_path data is written to
        :type sim_path: str
        :param start_year: start year of simulation
        :type start_year: int
        :param end_year: end year of simulation
        :type end_year: int
        :param sim_name: name of simulation
        :type sim_name: str
        :param dependency: sim_name of simulation to depend on
        :type dependency: str
        :param write_output: output ids of `outputs` to be written by
            LPJmL. Make sure to check if required output is available via
            `get_output_avail`
        :type write_output: list
        :param write_temporal_resolution: list of temporal resolutions
            corresponding to `outputs` or str to set the same resolution for
            all `outputs`. Choose between "annual", "monthly", "daily".
            Defaults to "annual" (use default output/outputvar resolution).
        :type write_temporal_resolution: list/str
        :param write_file_format: file format of output files. Choose between
            "raw", "clm" and "cdf". Defaults to "cdf".
        :type write_file_format: str
        :param append_output: if True defined output entries are appended by
            defined `outputs`. Please mind that the existing ones are not
            altered.
        :param append_output: bool
        """
        self.sim_name = sim_name
        self.sim_path = create_subdirs(sim_path, self.sim_name)
        output_path = f"{sim_path}/output/{self.sim_name}"
        # set time range for historic run
        self._set_timerange(start_year=start_year,
                            end_year=end_year,
                            write_start_year=start_year)
        # set output writing
        self._set_output(output_path,
                         outputs=write_output,
                         temporal_resolution=write_temporal_resolution,
                         file_format=write_file_format,
                         append_output=append_output)
        # set start from directory to start from spinup run
        self._set_startfrom(path=f"{sim_path}/restart",
                            dependency=dependency)
        # set restart directory to restart from in subsequent transient run
        self._set_restart(path=f"{sim_path}/restart")

    def set_coupled(self,
                    sim_path,
                    start_year, end_year,
                    coupled_input, coupled_output,
                    sim_name="coupled",
                    dependency = None,
                    coupled_year=None,
                    write_output=[],
                    write_temporal_resolution="annual",
                    write_file_format="cdf",
                    append_output=True,
                    model_name="copan:CORE"):
        """Set configuration required for coupled model runs
        :param sim_path: define sim_path data is written to
        :type sim_path: str
        :param start_year: start year of simulation
        :type start_year int
        :param end_year: end year of simulation
        :type end_year: int
        :param coupled_input: list of inputs to be used as socket for coupling.
            Provide dictionary/json key as identifier -> entry in list.
        :type coupled_input: list
        :param coupled_output: list of outputs to be used as socket for
            coupling. Provide output id as identifier -> entry in list.
        :type coupled_output: list
        :param sim_name: name of simulation
        :type sim_name: str
        :param dependency: sim_name of simulation to depend on
        :type dependency: str
        :param coupled_year: start year of coupled simulation
        :type coupled_year: int/None
        :param write_output: output ids of `outputs` to be written by
            LPJmL. Make sure to check if required output is available via
            `get_output_avail`
        :type write_output: list
        :param write_temporal_resolution: list of temporal resolutions
            corresponding to `outputs` or str to set the same resolution for
            all `outputs`. Choose between "annual", "monthly", "daily".
            Defaults to "annual" (use default output/outputvar resolution).
        :type write_temporal_resolution: list/str
        :param write_file_format: file format of output files. Choose between
            "raw", "clm" and "cdf". Defaults to "cdf".
        :type write_file_format: str
        :param append_output: if True defined output entries are appended by
            defined `outputs`. Please mind that the existing ones are not
            altered.
        :param append_output: bool
        :param model_name: model name of the coupled program which also sets
            the model to coupled mode (without coupling coupled_model = None)
        :type model_name: str
        """
        self.sim_name = sim_name
        self.sim_path = create_subdirs(sim_path, self.sim_name)
        output_path = f"{sim_path}/output/{self.sim_name}"

        # set time range for coupled run
        self._set_timerange(start_year=start_year,
                            end_year=end_year,
                            write_start_year=start_year)
        # set grid explicitly to be able to use start and endgrid
        self._set_grid_explicitly()

        # set output directory, outputs (relevant ones for pbs and agriculture)
        write_output += [
            item for item in coupled_output if item not in write_output
        ]
        self._set_output(output_path,
                         outputs=write_output,
                         temporal_resolution=write_temporal_resolution,
                         file_format=write_file_format,
                         append_output=append_output)
        # set coupling parameters
        self._set_coupling(inputs=coupled_input,
                           outputs=coupled_output,
                           start_year=coupled_year,
                           model_name=model_name)
        # set start from directory to start from historic run
        self._set_startfrom(path=f"{sim_path}/restart",
                            dependency=dependency)

    def _set_output(self, output_path, outputs=[], file_format="raw",
                    temporal_resolution="annual", append_output=True):
        """Set outputs to be written by LPJmL, define temporal resolution
        :param output_path: define output_path the output is written to. If
            `append_output == True` output_path is only altered for appended
            `outputs`.
        :type output_path: str
        :param outputs: output ids of `outputs` to be written by LPJmL. Make
            sure to check if required output is available via
            `get_output_avail`
        :type outputs: list
        :param file_format: file format for `outputs` (not to be used for
            sockets!). "raw" (binary), "clm" (binary with header) and "cdf"
            (NetCDF) are availble. Defaults to "raw".
        :type file_format: str
        :param temporal_resolution: list of temporal resolutions corresponding
            to `outputs` or str to set the same resolution for all `outputs`.
            Defaults to "annual" (for all `outputs`).
        :type temporal_resolution: list/str
        :param append_output: if True defined output entries are appended by
            defined `outputs`. Please mind that the existing ones are not
            altered.
        :param append_output: bool
        """
        available_res = ('annual', 'monthly', 'daily')
        available_formats = {'raw': 'bin', 'clm': 'clm', 'cdf': 'nc4'}
        nonvariable_outputs = ('globalflux')

        # provide additional meta data
        self.output_metafile = True

        # add grid output if not already defined
        if "grid" not in outputs:
            outputs.append("grid")
            if isinstance(temporal_resolution, list) > 1:
                temporal_resolution.append("annual")

        if temporal_resolution:
            # check if temporal resolution is of length one
            if isinstance(temporal_resolution, list):
                temp_res = temporal_resolution
                if len(outputs) != len(temporal_resolution):
                    raise ValueError("outputs and temporal_resolution have a" +
                                     " different length. Please adjust.")
            else:
                temp_res = [temporal_resolution]

            # check if temporal resolution is available
            if any(res not in available_res for res in temp_res):
                raise ValueError("Temporal resolution not available for " +
                                 "LPJmL. Choose from 'annual', 'monthly' " +
                                 "and 'daily'.")

        # create dict of outputvar names with indexes for iteration
        outputvar_names = {
            ov.name: pos for pos, ov in enumerate(self.outputvar)
        }
        # extract dict of outputvar for manipulation
        outputvars = self.to_dict()['outputvar']

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
                if out.id not in nonvariable_outputs:
                    self.output[pos].file.fmt = file_format
                    self.output[pos].file.name = (
                        f"{output_path}/{out.id}"
                        f".{available_formats[file_format]}"
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
                if not temporal_resolution:
                    timestep = outputvars[outputvar_names[out]]['timestep']
                elif isinstance(temporal_resolution, list):
                    timestep = temporal_resolution[pos]
                else:
                    timestep = temporal_resolution

                # create new output entry
                new_out = SubConfig({
                    'id': outputvars[outputvar_names[out]]['name'],
                    'file': SubConfig({
                        'fmt': file_format,
                        # 'socket': False,
                        'timestep': timestep,
                        'name': f"{output_path}/"
                                f"{outputvars[outputvar_names[out]]['name']}"
                                f".{available_formats[file_format]}",
                    })
                })
                self.output.append(new_out)
            elif out not in outputvar_names:
                # raise error if defined outputs are not in outputvar
                raise ValueError(
                    f"The following output is not defined in outputvar: {out}"
                )

    def set_outputpath(self, output_path):
        """Set output path of specified outputs
        :param output_path: path for outputs to be written, could also b
            relative path
        :type output_path: str
        """
        for out in self.output:
            file_name = out.file.name.split("/")
            file_name.reverse()
            out.file.name = f"{output_path}/{file_name[0]}"

    def _set_startfrom(self, path, dependency=None):
        """Set restart file from which LPJmL starts the transient run
        """
        if dependency:
            self.restart_filename = (
                f"{path}/restart_{dependency}.lpj")

        else:
            file_name = self.restart_filename.split("/")
            file_name.reverse()
            self.restart_filename = f"{path}/{file_name[0]}"

    def _set_restart(self, path):
        """Set restart file from which LPJmL starts the transient run
        """
        self.write_restart_filename = (
            f"{path}/restart_{self.sim_name}.lpj")
        self.restart_year = self.lastyear

    def _set_timerange(self,
                       start_year=1901,
                       end_year=2017,
                       write_start_year=None):
        """Set simulation time range, outputyear to start as a default here.
        :param start_year: start year of simulation
        :type start_year: int
        :param end_year: end year of simulation
        :type end_year: int
        :param write_start_year: first year of output being written
        :type write_start_year: int
        """
        self.firstyear = start_year
        self.lastyear = end_year
        if write_start_year:
            self.outputyear = write_start_year
        else:
            self.outputyear = start_year

    def _set_grid_explicitly(self, only_all=True):
        """Set startgrid and endgrid for LPJmL simulation
        """
        if self.startgrid == "all":
            self.startgrid = 0
        if self.endgrid == "all" or not only_all:
            if self.input.soil.fmt in ['json', 'meta']:
                self.endgrid = read_config(
                    f"{self.inpath}/{self.input.soil.name}",
                    to_dict=True
                )["ncell"] - 1
            else:
                if os.path.isfile(self.input.soil.name):
                    self.endgrid = os.path.getsize(
                        self.input.soil.name
                    ) - 1
                else:
                    self.endgrid = os.path.getsize(
                        f"{self.inpath}/{self.input.soil.name}"
                    ) - 1

    def _set_coupling(self,
                      inputs,
                      outputs,
                      start_year=None,
                      model_name="copan:CORE"):
        """Coupled settings - no spinup, not write restart file and set sockets
        for inputs and outputs (via corresponding ids)
        :param inputs: list of inputs to be used as socket for coupling.
            Provide dictionary/json key as identifier -> entry in list.
        :type inputs: list
        :param outputs: list of outputs to be used as socket for coupling.
            Provide output id as identifier -> entry in list.
        :type outputs: list
        :param start_year: start year of model coupling
        :type start_year: int
        :param model_name: model name of the coupled program which also sets
            the model to coupled mode (without coupling coupled_model = None)
        :type model_name: str
        """
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
        """Set sockets for inputs and outputs (via corresponding ids)
        :param inputs: list of inputs to be used as socket for coupling.
            Provide dictionary/json key as identifier -> entry in list.
        :type inputs: list
        """
        for inp in inputs:
            sock_input = getattr(self.input, inp)
            if 'id' not in sock_input.__dict__.keys():
                raise ValueError('Please use a config with input ids.')
            sock_input.__dict__['socket'] = True

    def _set_outputsockets(self, outputs=[]):
        """Set sockets for inputs and outputs (via corresponding ids)

        :param outputs: list of outputs to be used as socket for coupling.
            Provide output id as identifier -> entry in list.
        :type outputs: list
        """
        if "grid" not in outputs:
            outputs.append("grid")

        # get names/ids only of outputs that are defined in outputvar
        valid_outs = {
            out.name for out in self.outputvar if out.name in outputs
        }

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
                self.output[pos].file.timestep = "annual"

    def get_input_sockets(self, id_only=False):
        """get defined socket inputs as dict
        """
        inputs = self.input.to_dict()
        if id_only:
            return [inp for inp in inputs if (
                "socket" in inputs[inp]) and inputs[inp]["socket"]]
        else:
            return {
                inp: inputs[inp] for inp in inputs if (
                    "socket" in inputs[inp]) and inputs[inp]["socket"]
            }

    def get_output_sockets(self, id_only=False):
        """get defined socket outputs as dict
        """
        outputs = self.to_dict()["output"]
        name_id = {out.name: out.id for out in self.outputvar}

        if id_only:
            return [out["id"] for out in outputs if (
                "socket" in out["file"]) and out["file"]["socket"]]
        else:
            return {
                out["id"]: dict(
                    {'index': name_id[out["id"]]}, **out
                ) for out in outputs if (
                    "socket" in out["file"]) and out["file"]["socket"]
            }

    def add_config(self, file_name):
        """Add config file of coupled model to LPJmL config
        :param file_name: path to coupled config file
        :type file_name: str
        """
        self.coupled_config = read_yaml(file_name, CoupledConfig)

    def regrid(self,
               sim_path,
               model_path=None,
               country_code='BEL',
               overwrite_input=False):
        """Regrid LPJmL configuration file to a new country.
        :param sim_path: directory to check wether required subfolders exists. If
            not create corresponding folder (input, output, restart)
        :type sim_path: str
        :param model_path: path to `LPJmL_internal` (lpjml repository)
        :type model_path: str
        :param country_code: country code of country to regrid to. Defaults to
            'LUX'.
        :type country_code: str
        :param overwrite_input: overwrite existing country specific input files.
            Defaults to False.
        :type overwrite_input: bool
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
            (countries[country]["name"] for country in countries if (
                countries[country]["code"] == country_code
            )),
            None
        ).lower()

        if not os.path.isfile(self.input.coord.name):
            grid_file = f"{self.inpath}/{self.input.coord.name}"
        else:
            grid_file = self.input.coord.name

        # check if country specific input files already exist
        if not os.path.isfile(f"{sim_path}/input/coord_{country}.clm") or (
            not os.path.isfile(f"{sim_path}/input/soil_{country}.bin")
        ) or (
            not os.path.isfile(f"{sim_path}/input/lakes_{country}.bin")
        ) or overwrite_input:

            # extract country specific grid
            run([
                f"{model_path}/bin/getcountry",
                f"{self.inpath}/{self.input.countrycode.name}",
                grid_file,
                f"{sim_path}/input/coord_{country}.clm",
                country_code
            ], check=True)

            # extract country specific soil file from meta file
            if self.input.soil.fmt in ['json', 'meta']:
                soil_filename = read_json(
                    f"{self.inpath}/{self.input.soil.name}",
                )["filename"]
                soil_file = f"{self.inpath}/{self.input.soil.name}"
                soil_file = (
                    f"{soil_file[:soil_file.rfind('/')+1]}{soil_filename}"
                )
            else:
                soil_file = f"{self.inpath}/{self.input.soil.name}"

            # regrid soil file to country specific grid
            run([
                f"{model_path}/bin/regridsoil",
                grid_file,
                f"{sim_path}/input/coord_{country}.clm",
                soil_file,
                f"{sim_path}/input/soil_{country}.bin",
            ], check=True, stdout=open(os.devnull, 'wb'))

            # extract country specific lakes file from meta file
            if self.input.lakes.fmt in ['json', 'meta']:
                lakes_filename = read_json(
                    f"{self.inpath}/{self.input.lakes.name}"
                )["filename"]
                lakes_file = f"{self.inpath}/{self.input.lakes.name}"
                lakes_file = (
                    f"{lakes_file[:lakes_file.rfind('/')+1]}{lakes_filename}"
                )
            else:
                lakes_file = f"{self.inpath}/{self.input.lakes.name}"

            # regrid lakes file to country specific grid
            run([
                f"{model_path}/bin/regridsoil",
                grid_file,
                f"{sim_path}/input/coord_{country}.clm",
                lakes_file,
                f"{sim_path}/input/lakes_{country}.bin",
            ], check=True, stdout=open(os.devnull, 'wb'))

            # loop over all used input files to regrid them to country specific
            #   grid
            for config_key, config_input in self.input:

                if config_input.fmt != 'clm' or config_key == 'coord' or (
                    config_input.name == 'DUMMYLOCATION'
                ):                    continue

                if os.path.isfile(config_input.name):
                    input_file = config_input.name
                else:
                    input_file = f"{self.inpath}/{config_input.name}"

                # for some input files an additional argument may be  required
                # if config_key in [
                #     'lightning',
                #     'no3_deposition',
                #     'nh4_deposition',
                #     'drainage',
                #     'neighb_irrig',
                #     'reservoir',
                #     'wateruse_1900_2000'
                # ]:
                #     additional_arg = "-size4"
                # else:
                #     additional_arg = ""

                # regrid all other input files to country specific grid
                regrid_cmd = [
                    f"{model_path}/bin/regridclm",
                    grid_file,
                    f"{sim_path}/input/coord_{country}.clm",
                    input_file,
                    f"{sim_path}/input/{config_key}_{country}.clm",
                ]
                # if additional_arg:
                #     regrid_cmd.insert(1, additional_arg)
                run(regrid_cmd, check=True, stdout=open(os.devnull, 'wb'))

        # change config file to country specific input files
        for config_key, config_input in self.input:
            if config_input.fmt not in ['clm', 'json', 'meta']:
                continue
            if config_key in ['soil', 'lakes']:
                config_input.fmt = "raw"
                config_input.name = (
                    f"{sim_path}/input/{config_key}_{country}.bin"
                )
            else:
                config_input.fmt = "clm"
                config_input.name = (
                    f"{sim_path}/input/{config_key}_{country}.clm"
                )

        self._set_grid_explicitly(only_all=False)

    def __repr__(self, sub_repr=0):
        """Representation of the config object
        """
        summary_attr = ["sim_id", "sim_name", "version",
                        "firstyear", "lastyear", "startgrid", "endgrid",
                        "landuse", "coupled_model", "start_coupling",
                        "coupled_config"]
        changed_repr = [
            to_repr for to_repr in self.changed if to_repr not in summary_attr
        ]
        if sub_repr > 0:
            spacing = "\n" + "  " * sub_repr
            summary = "Configuration:"
        else:
            summary = f"<pycoupler.{self.__class__.__name__}>"
            spacing = "\n"
        summary = spacing.join([
            summary,
            f"Settings:      {self.sim_id} v{self.version}",
            "  (general)",
            f"  * sim_name   {self.sim_name}",
            f"  * firstyear  {self.firstyear}",
            f"  * lastyear   {self.lastyear}",
            f"  * startgrid  {self.startgrid}",
            f"  * endgrid    {self.endgrid}",
            f"  * landuse    {self.landuse}",
        ])
        if changed_repr:
            summary_list = [summary, "  (changed)"]
            summary_list.extend([
                f"  * {torepr}{(20-len(torepr))*' '} {getattr(self, torepr)}"
                for torepr in changed_repr
            ])
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
            summary = spacing.join([
                summary,
                f"Coupled model:        {self.coupled_model}",
                f"  * start_coupling    {sc}",
                f"  * input (coupled)   {input_coupled}",
                f"  * output (coupled)  {output_coupled}",
                coupled_config_repr
            ])

        return summary

    def __setattr__(self, __name, __value):
        super().__setattr__(__name, __value)
        self.changed.append(__name)


def parse_config(file_name="./lpjml.js", spin_up=False,
                 macros=None, config_class=False):
    """Precompile lpjml.js and return LpjmlConfig object or dict. Also
    evaluate macros. Analogous to R function `lpjmlKit::parse_config`.
    :param path: path to lpjml root
    :type path: str
    :param js_filename: js file filename, defaults to lpjml.js
    :type js_filename: str
    :param spin_up: convenience argument to set macro whether to start
        from restart file (`True`) or not (`False`). Defaults to `True`
    :type spin_up: bool
    :param macros: provide a macro in the form of "-DMACRO" or list of macros
    :type macros: str, list
    :param config_class: class of config object to be returned or None
        (return dict)
    :type config_class: class
    :return: A LpjmlConfig object
    :rtype: LpjmlConfig, dict
    """
    # precompile command
    cmd = ["cpp", "-P"]
    # add arguments
    if not spin_up:
        cmd.append('-DFROM_RESTART')
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


def read_config(file_name,
                model_path=None,
                spin_up=False,
                macros=None,
                to_dict=False):
    """Read function for config files to be returned as LpjmlConfig object or
    alternatively dict.
    :param file_name: file name (including relative/absolute path) of the
        corresponding LPJmL configuration.
    :type file_name: str
    :param model_path: path to model root directory, defaults to None
    :type model_path: str, optional
    :param spin_up: convenience argument to set macro whether to start
        from restart file (`True`) or not (`False`). Defaults to `True`
    :type spin_up: bool, optional
    :param to_dict: if `True` an LpjmlConfig object is returned,
        else (`False`) a dictionary is returned
    :type to_dict: bool
    :return: if `to_dict == True` -> LpjmlConfig object, else a
        a dictionary
    :rtype: LpjmlConfig, dict
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
    """Class to handle coupled model configurations.
    """
    def __repr__(self, sub_repr=1, order=1):
        """Representation of the config object
        """
        spacing = "\n" + "  " * sub_repr
        if order == 1 and sub_repr == 1:
            summary = f"Coupled configuration:{spacing}"
        else:
            summary = spacing

        for key, value in self.__dict__.items():
            if isinstance(value, SubConfig):
                summary += f"""{'  ' * sub_repr}* {key}: {value.__repr__(
                    sub_repr + 1, order + 1
                )}""".strip() + spacing
            else:
                summary += f"{'  ' * sub_repr}* {key:<20} {value}".strip() +\
                    spacing

        return summary


def read_yaml(file_name, config_class):
    with open(file_name, 'r') as f:
        yaml_data = ruamel.yaml.safe_load(f)

    return from_yaml(yaml_data, config_class)


def from_yaml(yaml_data, config_class):
    if isinstance(yaml_data, dict):
        return config_class({k: from_yaml(v, config_class)
                             for k, v in yaml_data.items()})
    elif isinstance(yaml_data, list):
        return [from_yaml(v, config_class) for v in yaml_data]
    else:
        return yaml_data


def regrid(config_file,
           sim_path,
           model_path,
           country_code='LUX',
           overwrite_input=False,
           new_config_file=None):
    """Regrid LPJmL configuration file to a new country.
    :param config_file: path to LPJmL configuration file
    :type config_file: str
    :param sim_path: directory to check wether required subfolders exists. If
        not create corresponding folder (input, output, restart)
    :type sim_path: str
    :param model_path: path to `LPJmL_internal` (lpjml repository)
    :type model_path: str
    :param country_code: country code of country to regrid to. Defaults to
        'LUX'.
    :type country_code: str
    :param overwrite_input: overwrite existing country specific input files.
        Defaults to False.
    :type overwrite_input: bool
    :param new_config_file: path to new config file. If None, the original
        config file is overwritten. Defaults to None.
    :type new_config_file: str
    """

    if not os.path.isfile(config_file):
        raise OSError(f"File '{config_file}' does not exist.")

    config = read_config(config_file, model_path=model_path)

    config.regrid(sim_path=sim_path,
                  model_path=model_path,
                  country_code=country_code,
                  overwrite_input=overwrite_input)

    if new_config_file is not None:
        config.to_json(new_config_file)
    else:
        config.to_json(config_file)

    return None
