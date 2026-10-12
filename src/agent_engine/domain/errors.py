"""Domain failures. This module performs no I/O and imports no adapters.

Callers translate these exceptions into CLI status. They are not SQLite errors.
"""


class UnknownTenant(Exception):
    """Raised when the tenant directory or its engine.db does not exist."""


class TenantMismatch(Exception):
    """Raised when a path or the tenant.id row does not match the bound tenant."""


class TaskNotFound(Exception):
    """Raised when the bound tenant file has no row for the task id."""


class FenceRejected(Exception):
    """Raised when a commit or renew names a fencing token the database no longer holds."""


class LeaseNotHeld(Exception):
    """Raised when a lease is still owned and unexpired for another worker."""


class StaleContent(Exception):
    """Raised when file bytes do not match the expected preimage hash."""


class AmbiguousEdit(Exception):
    """Raised when the text to replace matches more than once."""


class VerificationFailed(Exception):
    """Raised when the external verifier rejects a finish proposal."""


class ToolTimeout(Exception):
    """Raised when a jailed process group exceeds its timeout and is killed."""
