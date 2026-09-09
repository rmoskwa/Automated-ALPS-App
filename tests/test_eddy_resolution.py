"""
Which eddy build the synB0 route runs, and how it recovers when the GPU build fails.

Background: FSL's bare ``eddy`` is a Python dispatcher that decides GPU-vs-CPU by
regex-matching ``nvidia-smi`` for ``CUDA Version:``. NVIDIA driver 610 renamed
that label (``CUDA UMD Version:``), so the dispatcher silently ran ``eddy_cpu``
on a machine whose ``eddy_cuda10.2`` worked. MRtrix3's ``dwifslpreproc`` never
had that problem because it picks the highest ``eddy_cuda*`` on PATH itself and
falls back to CPU only if the run fails. These tests pin the synB0 route to the
same behaviour.

The resolver walks a real PATH, so these tests build one from ``tmp_path``
rather than faking ``shutil.which``.
"""

from __future__ import annotations

import os
import stat

import pytest

from dti_alps.processing import commands
from dti_alps.processing.commands import check_fsl_available, resolve_eddy_binaries
from dti_alps.processing.messages import Log
from dti_alps.processing.pipeline import PipelineRunner, PipelineState
from tests.fakes import FakeToolRunner


@pytest.fixture
def path_with(tmp_path, monkeypatch):
    """Install a PATH made of one directory holding executables named ``*names``."""

    def _install(*names: str, extra_dir_names: tuple[str, ...] = ()):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        for name in names:
            exe = bin_dir / name
            exe.write_text("#!/bin/sh\n")
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        extra = tmp_path / "bin2"
        extra.mkdir(exist_ok=True)
        for name in extra_dir_names:
            exe = extra / name
            exe.write_text("#!/bin/sh\n")
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), str(extra)]))

    return _install


# --- resolve_eddy_binaries --------------------------------------------------


class TestResolveEddyBinaries:
    def test_nothing_installed(self, path_with):
        path_with()
        assert resolve_eddy_binaries() == (None, None)

    def test_versioned_cuda_build_is_found_without_fsl_dispatcher(self, path_with):
        """The driver-610 case: eddy_cuda10.2 works, FSL's `eddy` would pick CPU."""
        path_with("eddy", "eddy_cpu", "eddy_cuda10.2")
        assert resolve_eddy_binaries() == ("eddy_cuda10.2", "eddy_cpu")

    def test_highest_cuda_version_wins(self, path_with):
        path_with("eddy_cuda10.2", "eddy_cuda11.3", "eddy_cuda9.1", "eddy_cpu")
        cuda, _ = resolve_eddy_binaries()
        assert cuda == "eddy_cuda11.3"

    def test_cuda_versions_compared_numerically_not_lexically(self, path_with):
        path_with("eddy_cuda9.1", "eddy_cuda10.2")
        assert resolve_eddy_binaries()[0] == "eddy_cuda10.2"

    def test_explicit_eddy_cuda_softlink_overrides_versioned_builds(self, path_with):
        """Same override dwifslpreproc honours: a user-made `eddy_cuda` wins."""
        path_with("eddy_cuda", "eddy_cuda11.3")
        assert resolve_eddy_binaries()[0] == "eddy_cuda"

    def test_cuda_search_spans_every_path_entry(self, path_with):
        path_with("eddy_cpu", extra_dir_names=("eddy_cuda10.2",))
        assert resolve_eddy_binaries() == ("eddy_cuda10.2", "eddy_cpu")

    def test_non_executable_cuda_file_is_ignored(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "eddy_cuda10.2").write_text("not runnable")
        monkeypatch.setenv("PATH", str(bin_dir))
        assert resolve_eddy_binaries() == (None, None)

    def test_unparseable_cuda_suffix_is_ignored(self, path_with):
        path_with("eddy_cuda_failure_output.txt", "eddy_cpu")
        assert resolve_eddy_binaries() == (None, "eddy_cpu")

    @pytest.mark.parametrize(
        "installed, expected_cpu",
        [
            (("eddy_cpu", "eddy_openmp", "eddy"), "eddy_cpu"),
            (("eddy_openmp", "eddy"), "eddy_openmp"),
            (("eddy",), "eddy"),
        ],
    )
    def test_cpu_preference_order(self, path_with, installed, expected_cpu):
        path_with(*installed)
        assert resolve_eddy_binaries()[1] == expected_cpu


# --- preflight agrees with the run ------------------------------------------


class TestPreflightUsesTheSameResolver:
    FSL = ["flirt", "fnirt", "invwarp", "applywarp", "fslmaths"]

    def test_versioned_cuda_build_alone_satisfies_eddy(self, path_with):
        path_with(*self.FSL, "eddy_cuda10.2")
        ok, missing = check_fsl_available(use_synb0=True)
        assert ok and missing == []

    def test_eddy_cpu_alone_satisfies_eddy(self, path_with):
        path_with(*self.FSL, "eddy_cpu")
        ok, missing = check_fsl_available(use_synb0=True)
        assert ok and missing == []

    def test_no_eddy_build_is_reported(self, path_with):
        path_with(*self.FSL)
        assert check_fsl_available(use_synb0=True) == (False, ["eddy"])


# --- the synB0 route across the ToolRunner seam -----------------------------


def _synb0_state(tmp_path) -> PipelineState:
    outputs = tmp_path / "OUTPUTS"
    outputs.mkdir()
    (outputs / "topup_fieldcoef.nii.gz").write_bytes(b"")
    (outputs / "topup_movpar.txt").write_text("")
    (outputs / "acqparams.txt").write_text("0 1 0 0.05\n")
    state = PipelineState(
        dwi_path="/in/dwi.nii.gz",
        bvecs_path="/in/dwi.bvec",
        bvals_path="/in/dwi.bval",
        output_dir=str(tmp_path / "out"),
        output_prefix="sub",
        use_synb0=True,
        synb0_output_dir=str(outputs),
    )
    (tmp_path / "out").mkdir()
    state.setup_output_paths()
    return state


def _runner(state, fake):
    logs: list[str] = []

    def progress(msg):
        if isinstance(msg, Log):
            logs.append(msg.text)

    runner = PipelineRunner(state, progress_callback=progress, runner=fake)
    runner.logs = logs  # type: ignore[attr-defined]
    return runner


@pytest.fixture
def synb0_runner(tmp_path, monkeypatch):
    """A PipelineRunner whose synB0 stage passes its on-disk checks with given eddy builds."""

    def _build(fake: FakeToolRunner, cuda: str | None, cpu: str | None) -> PipelineRunner:
        monkeypatch.setattr(commands, "resolve_eddy_binaries", lambda: (cuda, cpu))
        # nibabel would read the (nonexistent) DWI to size the index file.
        import nibabel as nib

        class _Img:
            shape = (2, 2, 2, 3)

        monkeypatch.setattr(nib, "load", lambda _p: _Img())
        state = _synb0_state(tmp_path)
        # The stage checks eddy's main output exists; a fake writes nothing.
        with open(state.preprocessed_dwi_path, "wb"):
            pass
        return _runner(state, fake)

    return _build


def _eddy_calls(fake: FakeToolRunner) -> list[list[str]]:
    return [c for c in fake.calls if c[0] != "dwi2mask"]


def _is_eddy(name):
    return lambda cmd: cmd[0] == name


class TestSynb0RouteEddySelection:
    def test_cuda_build_runs_when_it_succeeds(self, synb0_runner):
        fake = FakeToolRunner()
        assert synb0_runner(fake, "eddy_cuda10.2", "eddy_cpu").run_eddy_with_synb0()
        assert [c[0] for c in _eddy_calls(fake)] == ["eddy_cuda10.2"]

    def test_cuda_failure_falls_back_to_cpu_build(self, synb0_runner):
        fake = FakeToolRunner().on(_is_eddy("eddy_cuda10.2"), returncode=1)
        runner = synb0_runner(fake, "eddy_cuda10.2", "eddy_cpu")
        assert runner.run_eddy_with_synb0()
        calls = _eddy_calls(fake)
        assert [c[0] for c in calls] == ["eddy_cuda10.2", "eddy_cpu"]
        # Both attempts carry identical eddy arguments.
        assert calls[0][1:] == calls[1][1:]
        assert any("retrying with CPU build eddy_cpu" in line for line in runner.logs)

    def test_both_builds_failing_fails_the_stage(self, synb0_runner):
        fake = FakeToolRunner().on(lambda cmd: cmd[0].startswith("eddy"), returncode=1)
        runner = synb0_runner(fake, "eddy_cuda10.2", "eddy_cpu")
        assert not runner.run_eddy_with_synb0()
        assert [c[0] for c in _eddy_calls(fake)] == ["eddy_cuda10.2", "eddy_cpu"]
        assert any(line.startswith("ERROR: eddy failed") for line in runner.logs)

    def test_no_cuda_build_goes_straight_to_cpu(self, synb0_runner):
        fake = FakeToolRunner()
        runner = synb0_runner(fake, None, "eddy_cpu")
        assert runner.run_eddy_with_synb0()
        assert [c[0] for c in _eddy_calls(fake)] == ["eddy_cpu"]
        assert any("No CUDA eddy build found" in line for line in runner.logs)

    def test_no_eddy_at_all_fails_without_running_anything(self, synb0_runner):
        fake = FakeToolRunner()
        runner = synb0_runner(fake, None, None)
        assert not runner.run_eddy_with_synb0()
        assert _eddy_calls(fake) == []
        assert any("no eddy executable" in line for line in runner.logs)

    def test_cancel_during_cuda_attempt_does_not_retry_on_cpu(self, synb0_runner):
        """A user cancel is not a CUDA failure; the CPU build must not start."""

        class _CancellingFake(FakeToolRunner):
            pipeline: PipelineRunner

            def run(self, cmd, *, on_line=None, cancel_check=None):
                if cmd[0] == "eddy_cuda10.2":
                    self.pipeline.cancelled = True
                return super().run(cmd, on_line=on_line, cancel_check=cancel_check)

        fake = _CancellingFake().on(_is_eddy("eddy_cuda10.2"), cancel=True)
        runner = synb0_runner(fake, "eddy_cuda10.2", "eddy_cpu")
        fake.pipeline = runner
        assert not runner.run_eddy_with_synb0()
        assert [c[0] for c in _eddy_calls(fake)] == ["eddy_cuda10.2"]
        assert not any("retrying" in line for line in runner.logs)
