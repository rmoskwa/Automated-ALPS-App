"""
MRtrix3 command builders for DTI-ALPS pipeline.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import PipelineState


def _append_options_from_dict(cmd: list[str], options: dict) -> None:
    """
    Append CLI options from a dictionary to command list.

    Parameters
    ----------
    cmd : list of str
        Command list to append to
    options : dict
        Dictionary of option_name -> value pairs.
        For flags (bool True), only the option name is added.
        For other values, both option name and value are added.
    """
    for option_name, value in options.items():
        if value is None:
            continue
        if value is True:
            # Flag option (no value)
            cmd.append(option_name)
        elif value is False:
            # Disabled flag, skip
            continue
        elif isinstance(value, str) and value.strip():
            # String value - add with potential space prefix for nested options
            if option_name in ("-eddy_options", "-topup_options"):
                cmd.extend([option_name, f" {value}"])
            else:
                cmd.extend([option_name, value])
        elif isinstance(value, int | float):
            cmd.extend([option_name, str(value)])


def build_dwidenoise_cmd(state: "PipelineState") -> list[str]:
    """
    Build dwidenoise command for thermal noise removal.

    Parameters
    ----------
    state : PipelineState
        Pipeline configuration

    Returns
    -------
    list of str
        Command and arguments
    """
    cmd = ["dwidenoise"]

    # Input DWI
    cmd.append(state.dwi_path)

    # Output denoised DWI
    cmd.append(state.denoised_dwi_path)

    # Append options from dict
    _append_options_from_dict(cmd, state.dwidenoise_options)

    return cmd


def build_mrdegibbs_cmd(state: "PipelineState") -> list[str]:
    """
    Build mrdegibbs command for Gibbs ringing removal.

    Parameters
    ----------
    state : PipelineState
        Pipeline configuration

    Returns
    -------
    list of str
        Command and arguments
    """
    cmd = ["mrdegibbs"]

    # Input: use denoised output if denoising was run, otherwise raw DWI
    if state.denoised_dwi_path and state.run_denoising:
        cmd.append(state.denoised_dwi_path)
    else:
        cmd.append(state.dwi_path)

    # Output degibbs DWI
    cmd.append(state.degibbs_dwi_path)

    # Append options from dict
    _append_options_from_dict(cmd, state.mrdegibbs_options)

    return cmd


def build_dwi2mask_cmd(
    dwi_path: str, mask_path: str, bvecs_path: str | None = None, bvals_path: str | None = None
) -> list[str]:
    """
    Build dwi2mask command for brain mask generation.

    Parameters
    ----------
    dwi_path : str
        Path to DWI image
    mask_path : str
        Output path for brain mask
    bvecs_path : str, optional
        Path to bvecs file (required if not embedded in image)
    bvals_path : str, optional
        Path to bvals file (required if not embedded in image)

    Returns
    -------
    list of str
        Command and arguments
    """
    cmd = ["dwi2mask"]

    # Add gradient table if provided
    if bvecs_path and bvals_path:
        cmd.extend(["-fslgrad", bvecs_path, bvals_path])

    cmd.extend([dwi_path, mask_path])
    return cmd


def build_dwifslpreproc_cmd(state: "PipelineState") -> list[str]:
    """
    Build dwifslpreproc command for DWI preprocessing.

    Parameters
    ----------
    state : PipelineState
        Pipeline configuration

    Returns
    -------
    list of str
        Command and arguments
    """
    cmd = ["dwifslpreproc"]

    # Input: use degibbs output if available, then denoised, otherwise raw DWI
    if state.degibbs_dwi_path and state.run_degibbs:
        input_dwi = state.degibbs_dwi_path
    elif state.denoised_dwi_path and state.run_denoising:
        input_dwi = state.denoised_dwi_path
    else:
        input_dwi = state.dwi_path

    cmd.append(input_dwi)
    cmd.append(state.preprocessed_dwi_path)

    # Gradient table (FSL format)
    cmd.extend(["-fslgrad", state.bvecs_path, state.bvals_path])

    # Export corrected gradients
    bvecs_out = state.get_output_path("bvecs_preproc")
    bvals_out = state.get_output_path("bvals_preproc")
    cmd.extend(["-export_grad_fsl", bvecs_out, bvals_out])

    # Phase encoding direction
    cmd.extend(["-pe_dir", state.pe_direction])

    # Readout time
    cmd.extend(["-readout_time", str(state.readout_time)])

    # RPE scheme
    if state.rpe_scheme == "none":
        cmd.append("-rpe_none")
    elif state.rpe_scheme == "pair":
        cmd.append("-rpe_pair")
        if state.reverse_pe_path:
            cmd.extend(["-se_epi", state.reverse_pe_path])
            cmd.append("-align_seepi")
    elif state.rpe_scheme == "all":
        cmd.append("-rpe_all")
    elif state.rpe_scheme == "header":
        cmd.append("-rpe_header")
        # Only use -json_import with -rpe_header, as it relies on header/JSON for PE info
        # Using -json_import with explicit -pe_dir/-readout_time can cause conflicts
        if state.json_sidecar_path:
            cmd.extend(["-json_import", state.json_sidecar_path])

    # Legacy options (for backward compatibility)
    if state.eddy_mask_path:
        cmd.extend(["-eddy_mask", state.eddy_mask_path])
    if state.eddy_slspec_path:
        cmd.extend(["-eddy_slspec", state.eddy_slspec_path])

    # Append options from dict (new GUI options)
    _append_options_from_dict(cmd, state.dwifslpreproc_options)

    return cmd


def build_dwi2tensor_cmd(state: "PipelineState") -> list[str]:
    """
    Build dwi2tensor command for DTI fitting.

    Parameters
    ----------
    state : PipelineState
        Pipeline configuration

    Returns
    -------
    list of str
        Command and arguments
    """
    cmd = ["dwi2tensor"]

    # Input (preprocessed DWI)
    cmd.append(state.preprocessed_dwi_path)

    # Output tensor
    cmd.append(state.tensor_path)

    # Gradient table (use exported corrected gradients)
    bvecs_preproc = state.get_output_path("bvecs_preproc")
    bvals_preproc = state.get_output_path("bvals_preproc")
    cmd.extend(["-fslgrad", bvecs_preproc, bvals_preproc])

    # Legacy: Processing mask
    if state.dti_mask_path:
        cmd.extend(["-mask", state.dti_mask_path])

    # Append options from dict (new GUI options)
    _append_options_from_dict(cmd, state.dwi2tensor_options)

    return cmd


def build_tensor2metric_cmd(state: "PipelineState") -> list[str]:
    """
    Build tensor2metric command to extract FA and V1.

    Parameters
    ----------
    state : PipelineState
        Pipeline configuration

    Returns
    -------
    list of str
        Command and arguments
    """
    cmd = ["tensor2metric"]

    # Input tensor
    cmd.append(state.tensor_path)

    # Output FA (always required for ROI detection)
    cmd.extend(["-fa", state.fa_path])

    # Output principal eigenvector V1 (always required for fiber classification)
    cmd.extend(["-vector", state.v1_path])

    # Default V1 settings (can be overridden by options dict)
    # Check if options dict overrides -num or -modulate
    options = state.tensor2metric_options
    if "-num" not in options:
        cmd.extend(["-num", "1"])
    if "-modulate" not in options:
        cmd.extend(["-modulate", "none"])

    # Append options from dict (new GUI options)
    _append_options_from_dict(cmd, options)

    return cmd


def build_tensor2metric_alps_pas_cmds(state: "PipelineState") -> list[list[str]]:
    """
    Build tensor2metric commands to extract L1, L2, L3, V2, V3 for ALPS-PAS method.

    The ALPS-PAS method uses eigenvalues (L2, L3) sorted by eigenvector X-alignment
    rather than raw tensor diagonal components (Dxx, Dyy, Dzz). L1 is also extracted
    for completeness.

    ``tensor2metric``'s ``-num`` selects the eigenvalue/eigenvector index for every
    output flag in the same invocation, so the eigenvalue and eigenvector that share
    an index are emitted from one command instead of two. This collapses the former
    five invocations (each re-reading the tensor) into three -- one tensor read per
    index -- while producing byte-identical output files at the same paths. L1 keeps
    its own command because no L1 eigenvector is needed here.

    Parameters
    ----------
    state : PipelineState
        Pipeline configuration

    Returns
    -------
    list of list of str
        List of commands to execute (L1; L2+V2; L3+V3).
    """
    return [
        # First eigenvalue (L1) -- no matching eigenvector needed.
        ["tensor2metric", state.tensor_path, "-value", state.l1_path, "-num", "1"],
        # Second eigenvalue (L2) and eigenvector (V2, unmodulated) share -num 2.
        [
            "tensor2metric",
            state.tensor_path,
            "-value",
            state.l2_path,
            "-vector",
            state.v2_path,
            "-num",
            "2",
            "-modulate",
            "none",
        ],
        # Third eigenvalue (L3) and eigenvector (V3, unmodulated) share -num 3.
        [
            "tensor2metric",
            state.tensor_path,
            "-value",
            state.l3_path,
            "-vector",
            state.v3_path,
            "-num",
            "3",
            "-modulate",
            "none",
        ],
    ]


# --- Preflight --------------------------------------------------------------
# What the engine actually invokes, so a missing tool surfaces in the first
# second rather than three hours into stage 7. The lists below are the commands
# this codebase issues *directly* -- not the ones a wrapper calls internally --
# because those are what must resolve on PATH for our own subprocess calls.

# MRtrix3 commands issued by pipeline.py, b0_extraction.py, and the registration
# backend, on the standard (dwifslpreproc) route.
_MRTRIX_COMMANDS = [
    "dwidenoise",
    "mrdegibbs",
    "dwifslpreproc",
    "dwi2tensor",
    "tensor2metric",
    "dwi2mask",
    "dwiextract",
    "mrmath",
    "mrconvert",
]

# On the synB0 route the user has already run synB0-DISCO externally, so
# dwifslpreproc is never invoked -- `eddy` is called directly instead.
_MRTRIX_ONLY_STANDARD = {"dwifslpreproc"}

# FSL commands this codebase invokes directly: registration (FLIRT/FNIRT/INVWARP/
# APPLYWARP) and the FA masking step. `eddy`/`topup`/`applytopup` are *not* here
# on the standard route -- dwifslpreproc calls those itself, and requiring them
# would fail a perfectly good install where dwifslpreproc finds them by its own
# means.
_FSL_COMMANDS = ["flirt", "fnirt", "invwarp", "applywarp", "fslmaths"]

# On the synB0 route the pipeline runs `eddy` itself, so it must resolve for us.
_FSL_SYNB0_EXTRA = ["eddy"]

# Naming variants a single FSL command may legitimately have on PATH. Each entry
# is a template applied to the command name -- crucially *per command*, so a
# variant of one command can never satisfy another. (The previous list included
# the literal "eddy_openmp" for every command, which reported `topup` and
# `applytopup` present whenever `eddy_openmp` was installed.)
_FSL_VARIANT_TEMPLATES = ["{cmd}", "fsl{cmd}", "{cmd}_cuda", "{cmd}_openmp"]


def _find_any(variants: list[str]) -> bool:
    """True if any of ``variants`` resolves on PATH."""
    import shutil

    return any(shutil.which(variant) is not None for variant in variants)


# CPU eddy builds, in preference order. FSL >= 6.0.6 ships ``eddy_cpu``; older
# releases shipped ``eddy_openmp``. Bare ``eddy`` is FSL's own Python dispatcher,
# which may itself pick a GPU build -- it is the last resort, not a CPU guarantee.
_EDDY_CPU_CANDIDATES = ["eddy_cpu", "eddy_openmp", "eddy"]


def resolve_eddy_binaries() -> tuple[str | None, str | None]:
    """
    Return ``(cuda, cpu)``: the eddy programs on PATH to try, in that order.

    Mirrors MRtrix3's ``dwifslpreproc`` selection so the two preprocessing routes
    make the same choice on the same machine:

    - CUDA: an explicit ``eddy_cuda`` (a user's soft-link override) wins; otherwise
      the highest-versioned ``eddy_cudaX.Y`` anywhere on PATH.
    - CPU: the first of ``eddy_cpu``, ``eddy_openmp``, ``eddy``.

    We deliberately do *not* defer to FSL's ``eddy`` wrapper for the GPU decision.
    It parses ``nvidia-smi`` for a ``CUDA Version:`` label, and NVIDIA driver
    releases have changed that label, at which point the wrapper silently falls
    back to the CPU build on a machine whose GPU build runs fine. Selecting the
    binary ourselves and letting the run itself prove the GPU works is what
    ``dwifslpreproc`` does; either value may be ``None`` when nothing resolves.
    """
    import os
    import shutil

    cuda: str | None = None
    if shutil.which("eddy_cuda") is not None:
        cuda = "eddy_cuda"
    else:
        best_version = -1.0
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if not os.path.isdir(directory):
                continue
            for entry in os.listdir(directory):
                if not entry.startswith("eddy_cuda"):
                    continue
                try:
                    version = float(entry[len("eddy_cuda") :])
                except ValueError:
                    continue
                if version > best_version and shutil.which(entry) is not None:
                    best_version, cuda = version, entry

    cpu = next((c for c in _EDDY_CPU_CANDIDATES if shutil.which(c) is not None), None)
    return cuda, cpu


def check_mrtrix3_available(use_synb0: bool = False) -> tuple[bool, list[str]]:
    """
    Check that the MRtrix3 commands this engine invokes are on PATH.

    Parameters
    ----------
    use_synb0 : bool
        When True, check the synB0 route's requirements: ``dwifslpreproc`` is
        not invoked, because the user supplies synB0-DISCO's topup outputs and
        the pipeline runs ``eddy`` directly.

    Returns
    -------
    tuple of (bool, list)
        (all_available, list of missing commands)
    """
    import shutil

    required = [cmd for cmd in _MRTRIX_COMMANDS if not (use_synb0 and cmd in _MRTRIX_ONLY_STANDARD)]
    missing = [cmd for cmd in required if shutil.which(cmd) is None]
    return (len(missing) == 0, missing)


def check_fsl_available(use_synb0: bool = False) -> tuple[bool, list[str]]:
    """
    Check that the FSL commands this engine invokes are on PATH.

    Only commands issued *directly* are checked. ``topup`` and ``applytopup``
    are deliberately absent from the standard route: ``dwifslpreproc`` invokes
    those itself, and demanding them here would fail an install that works.

    Parameters
    ----------
    use_synb0 : bool
        When True, ``eddy`` is added -- on the synB0 route the pipeline runs it
        rather than delegating to ``dwifslpreproc``.

    Returns
    -------
    tuple of (bool, list)
        (all_available, list of missing commands)
    """
    required = _FSL_COMMANDS + (_FSL_SYNB0_EXTRA if use_synb0 else [])

    def _present(cmd: str) -> bool:
        # eddy is the one command we choose a build for ourselves, so preflight
        # must accept exactly what the run will accept (e.g. ``eddy_cuda10.2``).
        if cmd == "eddy":
            return any(resolve_eddy_binaries())
        return _find_any([tpl.format(cmd=cmd) for tpl in _FSL_VARIANT_TEMPLATES])

    missing = [cmd for cmd in required if not _present(cmd)]
    return (len(missing) == 0, missing)


def preflight(use_synb0: bool = False) -> list[str]:
    """
    Return the external commands this run needs but cannot find, in report order.

    The single answer to "will this run die on a missing tool?", so the two front
    ends ask the same question and cannot disagree about what counts as required.
    Both consume the list; only the phrasing differs (the CLI prints a report and
    exits 3, the GUI raises a readiness-strip row).

    Empty means every command the ``use_synb0`` route invokes resolves on PATH.
    """
    _, missing_mrtrix = check_mrtrix3_available(use_synb0)
    _, missing_fsl = check_fsl_available(use_synb0)
    return missing_mrtrix + missing_fsl
