"""tessera.quantum.holography — emergent spectral dimension from a Schwinger TDVP state.

The entire compute pipeline lives in the C++ extension; this module is a
thin re-export shim that surfaces the bindings under their natural import
path.

The four user-facing classes are:

* :class:`HolographyConfig`        — extends :class:`tessera.quantum.TDVPConfig`
                                     with the σ-grid and the mutual-information
                                     cutoff
* :class:`MutualInformationProfile`— symmetric MI matrix on the (site, snap)
                                     label set
* :class:`EmergentGraph`           — weighted Laplacian + heat-kernel trace +
                                     Graphviz export
* :class:`EmergentSpectralDimension` — workflow class: bind a HolographyConfig,
                                       call :meth:`compute` for the full result

Plus the data class :class:`SpectralDimensionResult` and the
:class:`AmbjornLollFit` static utility.

Quickstart
----------

>>> from tessera.quantum import TDVPConfig
>>> from tessera.quantum.holography import HolographyConfig, EmergentSpectralDimension
>>> cfg = HolographyConfig()
>>> cfg.tdvp = TDVPConfig()
>>> cfg.tdvp.N = 10; cfg.tdvp.a = 1.0; cfg.tdvp.g = 1.0
>>> cfg.tdvp.m = 0.5; cfg.tdvp.L0 = 0.0
>>> cfg.tdvp.dmrgMaxBondDim = 32; cfg.tdvp.dmrgNSweeps = 10
>>> cfg.tdvp.i0 = 3; cfg.tdvp.d = 3
>>> cfg.tdvp.dt = 0.2; cfg.tdvp.T = 0.4
>>> cfg.tdvp.snapshotEvery = 1
>>> cfg.tdvp.maxBondDim = 60
>>> result = EmergentSpectralDimension(cfg).compute()
>>> result.dInfinity > 0
True
"""

# Every name re-exported from the C++ ``quantum.holography`` submodule —
# bound below when the subsystem is available, used for a build-aware error
# when it is not.
_EXPORTS = (
    "HolographyConfig", "MutualInformationProfile", "EmergentGraph",
    "AmbjornLollFit", "AmbjornLollFitResult", "SpectralDimensionResult",
    "EmergentSpectralDimension", "ChoiPropagator", "ChoiTDVPSettings",
    "SchwingerParams",
)

_UNAVAILABLE_MESSAGE = (
    "tessera.quantum.holography is unavailable: this build of tessera does "
    "not include the quantum subsystem. Enable it with a single command — "
    "`TESSERA_QUANTUM=1 pip install -e .`; see tessera.quantum for details."
)

try:
    from tessera import _tessera
    _holography = _tessera.quantum.holography
    _AVAILABLE = True
    _IMPORT_ERROR = None
except (ImportError, AttributeError) as exc:
    _holography = None
    _AVAILABLE = False
    _IMPORT_ERROR = exc


def is_available() -> bool:
    """Return ``True`` if this tessera build includes the holography subsystem.

    Equivalent to :func:`tessera.quantum.is_available` — holography is part
    of the same optional C++ component.
    """
    return _AVAILABLE


if _AVAILABLE:
    for _name in _EXPORTS:
        globals()[_name] = getattr(_holography, _name)
    del _name
else:
    def __getattr__(name):
        if name in _EXPORTS:
            raise ImportError(_UNAVAILABLE_MESSAGE) from _IMPORT_ERROR
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*_EXPORTS, "is_available"]
