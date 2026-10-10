"""The one place audit entries are written. See models.AuditLog.

Takes IDs and action names only, by signature: there is no free-text argument,
so a narrative, a party name or any decrypted field has no way in.
"""
from flask import request
from flask_login import current_user

from . import db
from .models import AuditLog


def audit(action, target_type=None, target_id=None, outcome="success", user=None):
    """Append one entry and commit it.

    Call after the route's own commit: this commits the session, so anything
    still pending would be committed with it. `user` defaults to the signed-in
    account; pass it for events where nobody is signed in yet (a login).
    """
    if user is None and current_user.is_authenticated:
        user = current_user
    db.session.add(AuditLog(
        user_id=user.id if user else None,
        username=user.username if user else None,
        role=user.role if user else None,
        action=action,
        target_type=target_type,
        target_id=target_id,
        outcome=outcome,
        ip=request.remote_addr,
    ))
    db.session.commit()
