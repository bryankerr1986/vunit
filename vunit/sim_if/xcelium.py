# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2014-2021, Lars Asplund lars.anders.asplund@gmail.com

"""
Interface for the Cadence Xcelium simulator
"""

from pathlib import Path
from os.path import relpath
import sys
import os
import subprocess
import logging
from ..exceptions import CompileError
from ..ostools import write_file, file_exists, simplify_path
from ..vhdl_standard import VHDL
from . import SimulatorInterface, run_command, ListOfStringOption, check_output
from .cds_file import CDSFile
from ..color_printer import NO_COLOR_PRINTER

LOGGER = logging.getLogger(__name__)


class XceliumInterface(  # pylint: disable=too-many-instance-attributes
    SimulatorInterface
):
    """
    Interface for the Cadence Xcelium simulator
    """

    name = "xcelium"
    supports_gui_flag = True
    package_users_depend_on_bodies = False

    compile_options = [
        ListOfStringOption("xcelium.xrun_vhdl_flags"),
        ListOfStringOption("xcelium.xrun_verilog_flags"),
    ]

    sim_options = [ListOfStringOption("xcelium.xrun_sim_flags")]

    @staticmethod
    def add_arguments(parser):
        """
        Add command line arguments
        """
        group = parser.add_argument_group(
            "Xcelium/Incisive", description="Xcelium/Incisive specific flags"
        )
        group.add_argument(
            "--cdslib",
            default=None,
            help="The cds.lib file to use. If not given, VUnit maintains its own cds.lib file.",
        )
        group.add_argument(
            "--hdlvar",
            default=None,
            help="The hdl.var file to use. If not given, VUnit does not use a hdl.var file.",
        )

    @classmethod
    def from_args(cls, args, output_path, **kwargs):
        """
        Create new instance from command line arguments object
        """
        return cls(
            prefix=cls.find_prefix(),
            output_path=output_path,
            log_level=args.log_level,
            gui=args.gui,
            cdslib=args.cdslib,
            hdlvar=args.hdlvar,
        )

    @classmethod
    def find_prefix_from_path(cls):
        """
        Find xcelium simulator from PATH environment variable
        """
        return cls.find_toolchain(["xrun"])

    @staticmethod
    def supports_vhdl_contexts():
        """
        Returns True when this simulator supports VHDL 2008 contexts
        """
        return True

    def __init__(  # pylint: disable=too-many-arguments
        self, prefix, output_path, gui=False, log_level=None, cdslib=None, hdlvar=None
    ):
        SimulatorInterface.__init__(self, output_path, gui)
        self._prefix = prefix
        self._libraries = []
        self._log_level = log_level
        if cdslib is None:
            self._cdslib = str((Path(output_path) / "cds.lib").resolve())
        else:
            self._cdslib = str(Path(cdslib).resolve())
        self._hdlvar = hdlvar
        self._cds_root_xrun = self.find_cds_root_xrun()
        self._create_cdslib()

    def find_cds_root_xrun(self):
        """
        Finds xrun cds root
        """
        return subprocess.check_output(
            [str(Path(self._prefix) / "cds_root"), "xrun"]
        ).splitlines()[0].decode()

    def find_cds_root_virtuoso(self):
        """
        Finds virtuoso cds root
        """
        try:
            return subprocess.check_output(
                [str(Path(self._prefix) / "cds_root"), "virtuoso"],
                stderr=subprocess.STDOUT
            ).splitlines()[0].decode()
        except subprocess.CalledProcessError:
            return None

    def _create_cdslib(self):
        """
        Create the cds.lib file in the output directory if it does not exist
        """
        cds_root_virtuoso = self.find_cds_root_virtuoso()

        if cds_root_virtuoso is None:
            contents = """\
## cds.lib: Defines the locations of compiled libraries.
softinclude {0}/tools/inca/files/cds.lib
# needed for referencing the library 'basic' for cells 'cds_alias', 'cds_thru' etc. in analog models:
# NOTE: 'virtuoso' executable not found!
# define basic ".../tools/dfII/etc/cdslib/basic"
define work "{1}/libraries/work"
""".format(
                self._cds_root_xrun, self._output_path
            )
        else:
            contents = """\
## cds.lib: Defines the locations of compiled libraries.
softinclude {0}/tools/inca/files/cds.lib
# needed for referencing the library 'basic' for cells 'cds_alias', 'cds_thru' etc. in analog models:
define basic "{1}/tools/dfII/etc/cdslib/basic"
define work "{2}/libraries/work"
""".format(
                self._cds_root_xrun, cds_root_virtuoso, self._output_path
            )
        write_file(self._cdslib, contents)

    def setup_library_mapping(self, project):
        """
        Compile project using vhdl_standard
        """
        mapped_libraries = self._get_mapped_libraries()

        for library in project.get_libraries():
            self._libraries.append(library)
            self.create_library(library.name, library.directory, mapped_libraries)

    def compile_source_file_command(self, source_file):
        """
        Returns the command to compile a single source file
        """
        if source_file.is_vhdl:
            return self.compile_vhdl_file_command(source_file)

        if source_file.is_any_verilog:
            return self.compile_verilog_file_command(source_file)

        LOGGER.error("Unknown file type: %s", source_file.file_type)
        raise CompileError

    def _compile_all_source_files(self, source_files_by_library, printer):
        """
        Compiles all source files and prints status information
        """
        try:
            command = self.compile_all_files_command(source_files_by_library)
        except CompileError:
            command = None
            printer.write("failed", fg="ri")
            printer.write("\n")
            printer.write(f"File type not supported by {self.name!s} simulator\n")

            return False

        try:
            output = check_output(command, env=self.get_env())
            printer.write("passed", fg="gi")
            printer.write("\n")
            printer.write(output)

        except subprocess.CalledProcessError as err:
            printer.write("failed", fg="ri")
            printer.write("\n")
            printer.write(f"=== Command used: ===\n{subprocess.list2cmdline(command)!s}\n")
            printer.write("\n")
            printer.write(f"=== Command output: ===\n{err.output!s}\n")

            return False

        return True

    def compile_source_files(
        self,
        project,
        printer=NO_COLOR_PRINTER,
        continue_on_error=False,
        target_files=None,
    ):
        """
        Use compile_source_file_command to compile all source_files
        param: target_files: Given a list of SourceFiles only these and dependent files are compiled
        """
        dependency_graph = project.create_dependency_graph()
        failures = []

        if target_files is None:
            source_files = project.get_files_in_compile_order(dependency_graph=dependency_graph)
        else:
            source_files = project.get_minimal_file_set_in_compile_order(target_files)

        source_files_to_skip = set()

        max_library_name = 0
        max_source_file_name = 0
        if source_files:
            max_library_name = max(len(source_file.library.name) for source_file in source_files)
            max_source_file_name = max(len(simplify_path(source_file.name)) for source_file in source_files)

        source_files_by_library = {}
        for source_file in source_files:
            if source_file.library in source_files_by_library:
                source_files_by_library[source_file.library].append(source_file)
            else:
                source_files_by_library[source_file.library] = [source_file]
        # import pprint
        # pprint.pprint(source_files_by_library)

        printer.write("Compiling all source files")
        sys.stdout.flush()
        if self._compile_all_source_files(source_files_by_library, printer):
            printer.write("All source files compiled!")
        else:
            printer.write("One or more source files failed to compile.")
        exit()

        for source_file in source_files:
            printer.write(
                f"Compiling into {(source_file.library.name + ':').ljust(max_library_name + 1)!s} "
                f"{simplify_path(source_file.name).ljust(max_source_file_name)!s} "
            )
            sys.stdout.flush()
            exit()
            if source_file in source_files_to_skip:
                printer.write("skipped", fg="rgi")
                printer.write("\n")
                continue

            if self._compile_source_file(source_file, printer):
                project.update(source_file)
            else:
                source_files_to_skip.update(dependency_graph.get_dependent([source_file]))
                failures.append(source_file)

                if not continue_on_error:
                    break

        if failures:
            printer.write("Compile failed\n", fg="ri")
            raise CompileError

        if source_files:
            printer.write("Compile passed\n", fg="gi")
        else:
            printer.write("Re-compile not needed\n")


    @staticmethod
    def _vhdl_std_opt(vhdl_standard):
        """
        Convert standard to format of xrun command line flag
        """
        if vhdl_standard == VHDL.STD_2002:
            return "-v200x -extv200x"

        if vhdl_standard == VHDL.STD_2008:
            return "-v200x -extv200x -inc_v200x_pkg"

        if vhdl_standard == VHDL.STD_1993:
            return "-v93"

        raise ValueError("Invalid VHDL standard %s" % vhdl_standard)

    def _compile_all_files_in_library_subcommand(self, library, source_files):
        """
        Return a command to compile all source files in a library
        """
        args = []
        args += ["-makelib %s" % library.directory]
        args += ['-xmlibdirname "%s"' % str(Path(library.directory).parent)]
        args += [
            '-log "%s"'
            % str(
                Path(self._output_path)
                / ("xrun_compile_library_%s.log" % library.name)
            )
        ]

        for source_file in source_files:
            args += ["-filemap %s" % source_file.name]

            if source_file.is_vhdl:
                args += ["%s" % self._vhdl_std_opt(source_file.get_vhdl_standard())]
                args += source_file.compile_options.get("xcelium.xrun_vhdl_flags", [])

            if source_file.is_any_verilog:
                args += source_file.compile_options.get("xcelium.xrun_verilog_flags", [])
                for include_dir in source_file.include_dirs:
                    args += ['-incdir "%s"' % include_dir]
                for key, value in source_file.defines.items():
                    args += ["-define %s=%s" % (key, value.replace('"', '\\"'))]

            args += ["-endfilemap"]

        args += ["-endlib"]
        argsfile = str(
            Path(self._output_path)
            / ("xrun_compile_library_%s.args" % library.name)
        )
        write_file(argsfile, "\n".join(args))
        return ["-f", argsfile]


    def compile_all_files_command(self, source_files):
        """
        Return a command to compile all source files
        """
        cmd = str(Path(self._prefix) / "xrun")
        args = []

        args += ["-compile"]
        args += ["-nocopyright"]
        args += ["-licqueue"]
        # "Ignored unexpected semicolon following SystemVerilog description keyword (endfunction)."
        args += ["-nowarn UEXPSC"]
        # "cds.lib Invalid path"
        args += ["-nowarn DLCPTH"]
        # "cds.lib Invalid environment variable ''."
        args += ["-nowarn DLCVAR"]
        args += ["-work work"]
        args += ['-cdslib "%s"' % self._cdslib]
        args += self._hdlvar_args()
        args += [
            '-log "%s"'
            % str(
                Path(self._output_path)
                / ("xrun_compile_all.log")
            )
        ]
        if not self._log_level == "debug":
            args += ["-quiet"]
        else:
            args += ["-messages"]
            args += ["-libverbose"]
        # for "disciplines.vams" etc.
        args += ['-incdir "%s/tools/spectre/etc/ahdl/"' % self._cds_root_xrun]

        for library, _source_files in source_files.items():
            args += self._compile_all_files_in_library_subcommand(library, _source_files)

        argsfile = str(
            Path(self._output_path)
            / ("xrun_compile_all.args")
        )
        write_file(argsfile, "\n".join(args))
        return [cmd, "-f", argsfile]

    def compile_vhdl_file_command(self, source_file):
        """
        Returns command to compile a VHDL file
        """
        cmd = str(Path(self._prefix) / "xrun")
        args = []
        args += ["-compile"]
        args += ["-nocopyright"]
        args += ["-licqueue"]
        args += ["-nowarn DLCPTH"]  # "cds.lib Invalid path"
        args += ["-nowarn DLCVAR"]  # "cds.lib Invalid environment variable ''."
        args += ["%s" % self._vhdl_std_opt(source_file.get_vhdl_standard())]
        args += ["-work work"]
        args += ['-cdslib "%s"' % self._cdslib]
        args += self._hdlvar_args()
        args += [
            '-log "%s"'
            % str(
                Path(self._output_path)
                / ("xrun_compile_vhdl_file_%s.log" % source_file.library.name)
            )
        ]
        if not self._log_level == "debug":
            args += ["-quiet"]
        else:
            args += ["-messages"]
            args += ["-libverbose"]
        args += source_file.compile_options.get("xcelium.xrun_vhdl_flags", [])
        args += ['-xmlibdirname "%s"' % str(Path(source_file.library.directory).parent)]
        args += ["-makelib %s" % source_file.library.directory]
        args += ['"%s"' % source_file.name]
        args += ["-endlib"]
        argsfile = str(
            Path(self._output_path)
            / ("xrun_compile_vhdl_file_%s.args" % source_file.library.name)
        )
        write_file(argsfile, "\n".join(args))
        return [cmd, "-f", argsfile]

    def compile_verilog_file_command(self, source_file):
        """
        Returns commands to compile a Verilog file
        """
        cmd = str(Path(self._prefix) / "xrun")
        args = []
        args += ["-compile"]
        args += ["-nocopyright"]
        args += ["-licqueue"]
        # "Ignored unexpected semicolon following SystemVerilog description keyword (endfunction)."
        args += ["-nowarn UEXPSC"]
        # "cds.lib Invalid path"
        args += ["-nowarn DLCPTH"]
        # "cds.lib Invalid environment variable ''."
        args += ["-nowarn DLCVAR"]
        args += ["-work work"]
        args += source_file.compile_options.get("xcelium.xrun_verilog_flags", [])
        args += ['-cdslib "%s"' % self._cdslib]
        args += self._hdlvar_args()
        args += [
            '-log "%s"'
            % str(
                Path(self._output_path)
                / ("xrun_compile_verilog_file_%s.log" % source_file.library.name)
            )
        ]
        if not self._log_level == "debug":
            args += ["-quiet"]
        else:
            args += ["-messages"]
            args += ["-libverbose"]
        for include_dir in source_file.include_dirs:
            args += ['-incdir "%s"' % include_dir]

        # for "disciplines.vams" etc.
        args += ['-incdir "%s/tools/spectre/etc/ahdl/"' % self._cds_root_xrun]

        for key, value in source_file.defines.items():
            args += ["-define %s=%s" % (key, value.replace('"', '\\"'))]
        args += ['-xmlibdirname "%s"' % str(Path(source_file.library.directory).parent)]
        args += ["-makelib %s" % source_file.library.name]
        args += ['"%s"' % source_file.name]
        args += ["-endlib"]
        argsfile = str(
            Path(self._output_path)
            / ("xrun_compile_verilog_file_%s.args" % source_file.library.name)
        )
        write_file(argsfile, "\n".join(args))
        return [cmd, "-f", argsfile]

    def create_library(self, library_name, library_path, mapped_libraries=None):
        """
        Create and map a library_name to library_path
        """
        mapped_libraries = mapped_libraries if mapped_libraries is not None else {}

        lpath = str(Path(library_path).resolve().parent)

        if not file_exists(lpath):
            os.makedirs(lpath)

        if (
            library_name in mapped_libraries
            and mapped_libraries[library_name] == library_path
        ):
            return

        cds = CDSFile.parse(self._cdslib)
        cds[library_name] = library_path
        cds.write(self._cdslib)

    def _get_mapped_libraries(self):
        """
        Get mapped libraries from cds.lib file
        """
        cds = CDSFile.parse(self._cdslib)
        return cds

    def simulate(  # pylint: disable=too-many-locals, too-many-branches
        self, output_path, test_suite_name, config, elaborate_only=False
    ):
        """
        Elaborates and Simulates with entity as top level using generics
        """

        script_path = str(Path(output_path) / self.name)
        launch_gui = self._gui is not False and not elaborate_only

        if elaborate_only:
            steps = ["elaborate"]
        else:
            steps = ["elaborate", "simulate"]

        for step in steps:
            cmd = str(Path(self._prefix) / "xrun")
            args = []
            if step == "elaborate":
                args += ["-elaborate"]
            args += ["-nocopyright"]
            args += ["-licqueue"]
            # args += ['-dumpstack']
            # args += ['-gdbsh']
            # args += ['-rebuild']
            # args += ['-gdb']
            # args += ['-gdbelab']
            args += ["-errormax 10"]
            args += ["-nowarn WRMNZD"]
            args += ["-nowarn DLCPTH"]  # "cds.lib Invalid path"
            args += ["-nowarn DLCVAR"]  # "cds.lib Invalid environment variable ''."
            args += [
                "-xmerror EVBBOL"
            ]  # promote to error: "bad boolean literal in generic association"
            args += [
                "-xmerror EVBSTR"
            ]  # promote to error: "bad string literal in generic association"
            args += [
                "-xmerror EVBNAT"
            ]  # promote to error: "bad natural literal in generic association"
            args += ["-work work"]
            args += [
                '-xmlibdirname "%s"' % (str(Path(self._output_path) / "libraries"))
            ]  # @TODO: ugly
            args += config.sim_options.get("xcelium.xrun_sim_flags", [])
            args += ['-cdslib "%s"' % self._cdslib]
            args += self._hdlvar_args()
            args += ['-log "%s"' % str(Path(script_path) / ("xrun_%s.log" % step))]
            if not self._log_level == "debug":
                args += ["-quiet"]
            else:
                args += ["-messages"]
                # args += ['-libverbose']
            args += self._generic_args(config.entity_name, config.generics)
            for library in self._libraries:
                args += ['-reflib "%s"' % library.directory]
            args += ['-input "@set intovf_severity_level ignore"']
            if config.sim_options.get("disable_ieee_warnings", False):
                args += [
                    '-input "@set pack_assert_off { std_logic_arith numeric_std }"'
                ]
            args += [
                '-input "@set assert_stop_level %s"' % config.vhdl_assert_stop_level
            ]
            if launch_gui:
                args += ["-access +rwc"]
                # args += ['-linedebug']
                args += ["-gui"]
            else:
                args += ["-access +r"]
                args += ['-input "@run"']

            if config.architecture_name is None:
                # we have a SystemVerilog toplevel:
                args += ["-top %s.%s:sv" % (config.library_name, config.entity_name)]
            else:
                # we have a VHDL toplevel:
                args += [
                    "-top %s.%s:%s"
                    % (
                        config.library_name,
                        config.entity_name,
                        config.architecture_name,
                    )
                ]
            argsfile = "%s/xrun_%s.args" % (script_path, step)
            write_file(argsfile, "\n".join(args))
            if not run_command(
                [cmd, "-f", relpath(argsfile, script_path)],
                cwd=script_path,
                env=self.get_env(),
            ):
                return False
        return True

    def _hdlvar_args(self):
        """
        Return hdlvar argument if available
        """
        if self._hdlvar is None:
            return []
        return ['-hdlvar "%s"' % self._hdlvar]

    @staticmethod
    def _generic_args(entity_name, generics):
        """
        Create xrun arguments for generics/parameters
        """
        args = []
        for name, value in generics.items():
            if _generic_needs_quoting(value):
                args += ['''-gpg "%s.%s => \\"%s\\""''' % (entity_name, name, value)]
            else:
                args += ['''-gpg "%s.%s => %s"''' % (entity_name, name, value)]
        return args


def _generic_needs_quoting(value):  # pylint: disable=missing-docstring
    return isinstance(value, (str, bool))
