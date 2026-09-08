"""Runner checks using fake Python, without training or touching real outputs."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunnerEnvironmentTests(unittest.TestCase):
    def run_fixture(self, directory, *, python_status=0, mode="--resume", kernel=None):
        root = Path(directory)
        shutil.copy2(ROOT / "run_all.sh", root / "run_all.sh")
        for group in ("empirical", "empirical/plotter", "simulations"):
            (root / "src" / group).mkdir(parents=True, exist_ok=True)
        (root / "src/empirical/example.ipynb").write_text("{}")
        (root / "output").mkdir()
        sentinel = root / "output/keep.txt"
        sentinel.write_text("existing results")
        (root / "bin").mkdir()
        python = root / "bin/python"
        python.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$IMV_TEST_COMMANDS"\n'
            f"exit {python_status}\n"
        )
        python.chmod(0o755)
        commands = root / "commands.txt"
        env = {
            **os.environ,
            "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
            "XDG_RUNTIME_DIR": str(root),
            "MPLCONFIGDIR": str(root / "mpl"),
            "IMV_FIGURE_DIR": str(root / "output/figures"),
            "IMV_TEST_COMMANDS": str(commands),
        }
        env.pop("IMV_KERNEL_NAME", None)
        if kernel is not None:
            env["IMV_KERNEL_NAME"] = kernel
        result = subprocess.run(
            ["bash", str(root / "run_all.sh"), mode],
            env=env, capture_output=True, text=True, timeout=30,
        )
        return result, sentinel, commands.read_text()

    def test_failed_environment_check_preserves_fresh_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result, sentinel, commands = self.run_fixture(
                directory, python_status=1, mode="--fresh"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Environment check failed", result.stderr)
            self.assertEqual(sentinel.read_text(), "existing results")
            self.assertNotIn("nbconvert", commands)

    def test_runner_uses_current_python_and_explicit_project_kernel(self):
        for kernel in (None, "custom-imv"):
            with self.subTest(kernel=kernel), tempfile.TemporaryDirectory() as directory:
                result, sentinel, commands = self.run_fixture(directory, kernel=kernel)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(sentinel.is_file())
                self.assertIn("-m nbconvert --execute", commands)
                self.assertIn(
                    f"--ExecutePreprocessor.kernel_name={kernel or 'imv-ml'}", commands
                )


if __name__ == "__main__":
    unittest.main()
