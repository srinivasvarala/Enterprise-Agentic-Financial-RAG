"""Errors that cross module boundaries. The ingestion job treats them differently."""


class ParseError(Exception):
    """The document cannot be parsed. Never retried; the message is shown to the admin."""


class TransientError(Exception):
    """A provider failed in a way that may succeed on retry (timeout, 5xx, rate limit).

    Adapters translate their client's errors into this so the core stays provider-agnostic.
    """


class TenantRequiredError(Exception):
    """A vector store call was made without a tenant id. Always a coding mistake."""
