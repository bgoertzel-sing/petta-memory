from __future__ import annotations

import contextlib
import io
from typing import Callable, Protocol

from .store import MemoryCluster, ValidationError


class _PeTTaRuntime(Protocol):
    def process_metta_string(self, program: str): ...


def make_petta_parse_checker(
    petta: _PeTTaRuntime | None = None,
    *,
    petta_path: str | None = None,
    suppress_output: bool = True,
) -> Callable[[MemoryCluster], None]:
    """Build a ``MediumMemoryStore`` parse-check hook backed by PeTTa.

    The store remains dependency-free by default: this helper imports and/or uses
    a PeTTa runtime only when a caller explicitly asks for the external check.
    The returned checker sends the canonicalized ``MemoryCluster`` text to
    ``PeTTa.process_metta_string`` and raises ``ValidationError`` if the runtime
    rejects it.  It does not execute live OmegaClaw integration or write back to
    the journal.
    """
    runtime = petta if petta is not None else _load_petta_runtime(petta_path=petta_path)

    def check(cluster: MemoryCluster) -> None:
        try:
            if suppress_output:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    runtime.process_metta_string(cluster.text)
            else:
                runtime.process_metta_string(cluster.text)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"PeTTa runtime rejected cluster {cluster.cluster_id}: {exc}") from exc

    return check


def _load_petta_runtime(*, petta_path: str | None = None) -> _PeTTaRuntime:
    try:
        from petta import PeTTa
    except Exception as exc:  # pragma: no cover - exercised only without optional runtime
        raise ValidationError(f"PeTTa runtime is not importable: {exc}") from exc
    try:
        kwargs = {"verbose": False}
        if petta_path is not None:
            kwargs["petta_path"] = petta_path
        return PeTTa(**kwargs)
    except TypeError:
        # Keep compatibility with older/local PeTTa constructors that may not
        # accept the same keyword set.
        if petta_path is not None:
            return PeTTa(petta_path=petta_path)
        return PeTTa()
    except Exception as exc:  # pragma: no cover - depends on optional runtime setup
        raise ValidationError(f"PeTTa runtime could not be initialized: {exc}") from exc
