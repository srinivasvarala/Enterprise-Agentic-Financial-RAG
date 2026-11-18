"""Seeds the default Tenant and Collection whose ids live in config."""

from sqlalchemy.orm import Session

from finrag.core.config import Settings
from finrag.storage.models import Collection, Tenant


def seed_defaults(session: Session, settings: Settings) -> None:
    """Idempotent: creates the default Tenant + Collection if missing."""
    if session.get(Tenant, settings.default_tenant_id) is None:
        session.add(Tenant(id=settings.default_tenant_id, name="Default"))
        session.flush()
    if session.get(Collection, settings.default_collection_id) is None:
        session.add(
            Collection(
                id=settings.default_collection_id,
                tenant_id=settings.default_tenant_id,
                name="Default",
            )
        )
        session.flush()
