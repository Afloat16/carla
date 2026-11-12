"""A selected CARLA replaces the paths derived from the previous selection."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "src"))
CONFIG = importlib.import_module("carla_agentic_tools.config")
SERVER = importlib.import_module("carla_agentic_tools.server")
NODE = shutil.which("node")


class RootSelectionTest(unittest.TestCase):
    """Exercise both setters against real configuration files and root markers."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.config_path = self.root / "config.env"
        environment = dict(os.environ)
        for key in (*CONFIG.CONFIG_KEYS, *CONFIG.DERIVED_KEYS):
            environment.pop(key, None)
        environment["CARLA_TOOLS_CONFIG"] = str(self.config_path)
        self.environment = environment
        self.environment_patch = patch.dict(os.environ, environment, clear=True)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.trees = {}
        for flavor in ("ue4", "ue5", "ue58", "release"):
            tree = self.root / flavor
            (tree / "PythonAPI/carla").mkdir(parents=True)
            if flavor == "ue4":
                marker = tree / "Unreal/CarlaUE4/CarlaUE4.uproject"
                marker.parent.mkdir(parents=True)
                marker.touch()
            elif flavor in ("ue5", "ue58"):
                (tree / "Unreal/CarlaUnreal").mkdir(parents=True)
                (tree / "CMakePresets.json").write_text("{}", encoding="utf-8")
                if flavor == "ue58":
                    (tree / "Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Autoware").mkdir(parents=True)
                    (tree / "CMake").mkdir()
                    (tree / "CMake/DLSS.cmake").touch()
            else:
                (tree / "CarlaUnreal.sh").touch()
            self.trees[flavor] = tree

    def set_paths(self, implementation, updates):
        """Invoke the actual public setter, including validation and persistence."""
        if implementation == "python":
            SERVER.set_config(updates)
            return
        result = subprocess.run(
            [NODE, "-e", "require('./lib/skills').setConfig(JSON.parse(process.argv[1]))", json.dumps(updates)],
            cwd=TOOLS, env=self.environment, capture_output=True, text=True,
            timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def implementations(self):
        """The Python implementation is always exercised; Node is optional."""
        return ("python", "node") if NODE else ("python",)

    def test_switch_replaces_all_previous_derived_roots(self):
        """Old source and package paths must not keep obsolete skills enabled."""
        for implementation in self.implementations():
            for flavor in ("ue58", "release", "ue4", "ue5", "ue58"):
                with self.subTest(implementation=implementation, flavor=flavor):
                    tree = self.trees[flavor]
                    self.set_paths(implementation, {"CARLA_ROOT": str(tree)})
                    selected = CONFIG.detect_carla(tree)["vars"]
                    actual = CONFIG.read_config(self.root)
                    self.assertEqual(actual, selected)

    def test_clearing_root_clears_its_derived_paths(self):
        """Clearing the one prompted key must also clear its automatic aliases."""
        for implementation in self.implementations():
            for flavor in ("ue58", "release"):
                with self.subTest(implementation=implementation, flavor=flavor):
                    self.set_paths(implementation, {"CARLA_ROOT": str(self.trees[flavor])})
                    self.set_paths(implementation, {"CARLA_ROOT": ""})
                    self.assertEqual(CONFIG.read_config(self.root), {})

    def test_unrelated_updates_keep_the_selected_root(self):
        """Changing another tool path must not clear the current CARLA."""
        scenic = self.root / "scenic"
        (scenic / "src/scenic").mkdir(parents=True)
        for implementation in self.implementations():
            with self.subTest(implementation=implementation):
                self.set_paths(implementation, {"CARLA_ROOT": str(self.trees["ue58"])})
                before = CONFIG.read_config(self.root)
                self.set_paths(implementation, {"SCENIC_ROOT": str(scenic)})
                self.assertEqual(CONFIG.read_config(self.root), {**before, "SCENIC_ROOT": str(scenic)})

    def test_invalid_replacement_leaves_existing_config_intact(self):
        """Validation still happens before any selected-path changes reach disk."""
        for implementation in self.implementations():
            with self.subTest(implementation=implementation):
                self.set_paths(implementation, {"CARLA_ROOT": str(self.trees["ue58"])})
                before = self.config_path.read_bytes()
                if implementation == "python":
                    with self.assertRaises(ValueError):
                        self.set_paths(implementation, {"CARLA_ROOT": str(self.root / "missing")})
                else:
                    with self.assertRaises(AssertionError):
                        self.set_paths(implementation, {"CARLA_ROOT": str(self.root / "missing")})
                self.assertEqual(self.config_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
