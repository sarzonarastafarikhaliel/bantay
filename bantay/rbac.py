from functools import wraps

from flask import flash, redirect, request, url_for
from flask_login import current_user


def role_required(*roles):
    """Restrict a view to the given roles. Pair with @login_required."""
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                flash("You don't have permission to do that.")
                return redirect(request.referrer or url_for("dashboard.index"))
            return fn(*args, **kwargs)
        return wrapped
    return decorator
