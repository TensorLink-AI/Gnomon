"""Embed provenance in both wheels and sdists without modifying the checkout."""

import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        package = Path(self.root) / "src" / "gnomon"
        helper = runpy.run_path(str(package / "build_info.py"))
        info = helper["source_build_info"](package)
        self._temporary = TemporaryDirectory(prefix="gnomon-build-")
        path = Path(self._temporary.name) / "_build_info.json"
        path.write_text(json.dumps(info, sort_keys=True) + "\n", encoding="utf-8")
        destination = "gnomon/_build_info.json" if self.target_name == "wheel" else "src/gnomon/_build_info.json"
        build_data["force_include"][str(path)] = destination

    def finalize(self, version, build_data, artifact_path):
        self._temporary.cleanup()
