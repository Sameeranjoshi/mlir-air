# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

config.name = "CHIPLET"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.chiplet_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

config.chiplet_tools_dir = os.path.join(config.chiplet_obj_root, "bin")
config.chiplet_libs_dir = os.path.join(config.chiplet_obj_root, "lib")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.chiplet_tools_dir, config.llvm_tools_dir]
tools = ["mlir-opt", "chiplet-opt"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
