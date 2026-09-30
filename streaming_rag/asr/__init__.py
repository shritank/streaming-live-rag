"""Optional speech front end (audio -> transcript events). The text pipeline
never imports this package."""
from .local_agreement import LocalAgreement

__all__ = ["LocalAgreement"]
