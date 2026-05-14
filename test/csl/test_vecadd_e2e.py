"""End-to-end test for the AIR -> CSL vecadd milestone.

Compiles a 1x1 vecadd AIR program through:
  1. air-opt -air-to-csl-dialect (AIR -> csl.*)
  2. air-opt -csl-to-csl-rt     (csl.* -> csl.* + csl_rt.* runtime seq)
  3. air-translate --emit-csl-rt (writes layout.csl, pe_program.csl, run.py)
  4. cslc --memcpy               (compiles layout.csl + pe_program.csl -> out/)
  5. cs_python run.py --check    (runs on CS-3, validates result)

This is the milestone's definition of done. When this test passes on a
machine with CS-3 hardware, the AIR -> CSL backend is functional end
to end.

See docs/superpowers/plans/2026-04-14-air-to-csl-vecadd-v2.md §Phase 7
and docs/superpowers/specs/2026-04-13-air-to-csl-vecadd-design.md §2.
"""

import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_MLIR = REPO_ROOT / "mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir"
AIR_OPT = REPO_ROOT / "install/bin/air-opt"
AIR_TRANSLATE = REPO_ROOT / "install/bin/air-translate"


def test_vecadd_end_to_end(tmp_path):
    assert SRC_MLIR.exists(), f"missing input AIR program: {SRC_MLIR}"
    assert AIR_OPT.exists(), f"missing air-opt binary: {AIR_OPT}"
    assert AIR_TRANSLATE.exists(), f"missing air-translate binary: {AIR_TRANSLATE}"

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    compiled = artifacts / "out"

    # Step 1: MLIR pipeline -> layout.csl + vecadd_pe.csl + run.py
    pipeline_cmd = (
        f"{AIR_OPT} {SRC_MLIR} -air-to-csl-dialect -csl-to-csl-rt | "
        f"{AIR_TRANSLATE} --emit-csl-rt --csl-output-dir={artifacts} -o /dev/null"
    )
    result = subprocess.run(pipeline_cmd, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"air-opt | air-translate failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    assert (artifacts / "layout.csl").exists(), "missing generated layout.csl"
    assert (artifacts / "vecadd_pe.csl").exists(), "missing generated vecadd_pe.csl"
    assert (artifacts / "run.py").exists(), "missing generated run.py"

    # Step 2: cslc --memcpy -> compiled out/ directory
    cslc_result = subprocess.run(
        [
            "cslc", "--arch=wse3", "layout.csl",
            "--fabric-dims=8,3", "--fabric-offsets=4,1",
            "-o", "out",
            "--memcpy", "--channels", "1",
        ],
        cwd=str(artifacts),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert cslc_result.returncode == 0, (
        f"cslc failed:\n"
        f"--- stdout ---\n{cslc_result.stdout}\n"
        f"--- stderr ---\n{cslc_result.stderr}"
    )
    assert compiled.exists(), f"cslc did not produce {compiled}"

    # Step 3: cs_python run.py --check -> should print PASS
    run_result = subprocess.run(
        ["cs_python", "run.py", "--name", "out", "--check"],
        cwd=str(artifacts),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if run_result.returncode != 0:
        raise AssertionError(
            f"run.py failed on CS-3:\n"
            f"--- stdout ---\n{run_result.stdout}\n"
            f"--- stderr ---\n{run_result.stderr}"
        )
    assert "PASS" in run_result.stdout, (
        f"run.py did not print PASS:\n"
        f"--- stdout ---\n{run_result.stdout}\n"
        f"--- stderr ---\n{run_result.stderr}"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            test_vecadd_end_to_end(tmp_path)
            print("PASS: test_vecadd_end_to_end")
        except AssertionError as e:
            print(f"FAIL: test_vecadd_end_to_end")
            print(str(e))
            exit(1)
