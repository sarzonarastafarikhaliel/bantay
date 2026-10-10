import hmac
import secrets
from datetime import datetime, timedelta

from flask import Blueprint, abort, render_template, redirect, url_for, request, flash, session
from flask_login import current_user, login_user, logout_user, login_required

from .. import db
from ..audit import audit
from ..models import AuditLog, User
from ..seed import LEGACY_DEFAULT_PASSWORD

auth_bp = Blueprint("auth", __name__)

MIN_PASSWORD_LENGTH = 10


def _csrf_token():
    """Per-session token for the change-password form.

    The app has no CSRF protection elsewhere (no Flask-WTF); this form gets its
    own because a forged password change is an account takeover. Stdlib only.
    """
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def _csrf_ok():
    expected = session.get("csrf_token")
    sent = request.form.get("csrf_token", "")
    # A missing session token must fail, not compare "" == "".
    return bool(expected) and hmac.compare_digest(sent, expected)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            audit("login", user=user)
            return redirect(url_for("dashboard.index"))
        # Logged against the account only when the name is a real account -
        # free text from the username box (a mistyped password, a resident's
        # name) is never written to the log.
        audit("login", outcome="failure", user=user)
        flash("Invalid username or password.")
    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    audit("logout")
    logout_user()
    return redirect(url_for("auth.login"))


@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        if not _csrf_ok():
            abort(400)
        current = request.form.get("current_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        if not current_user.check_password(current):
            error = "Current password is incorrect."
        elif len(new) < MIN_PASSWORD_LENGTH:
            error = f"New password must be at least {MIN_PASSWORD_LENGTH} characters."
        elif new != confirm:
            error = "New password and confirmation do not match."
        elif new == current:
            error = "New password must be different from the current one."
        elif new == LEGACY_DEFAULT_PASSWORD:
            error = "That is the old shared default password. Choose another."
        else:
            error = None
        if error:
            audit("password_change", outcome="failure")
            flash(error)
            return redirect(url_for("auth.change_password"))

        user = current_user._get_current_object()
        user.set_password(new)
        user.must_change_password = False
        db.session.commit()
        # Rotate: drop everything in the old session (CSRF token included) and
        # sign in again, so the browser gets a freshly issued session cookie.
        session.clear()
        login_user(user)
        audit("password_change", user=user)
        flash("Password changed.")
        return redirect(url_for("dashboard.index"))
    return render_template("change_password.html", csrf_token=_csrf_token(),
                           min_length=MIN_PASSWORD_LENGTH)


@auth_bp.route("/audit")
@login_required
def audit_log():
    """Read-only view of the latest audit entries. No route edits or deletes them."""
    if current_user.role != "admin":
        audit(request.endpoint, outcome="denied")
        abort(403)
    filters = {k: request.args.get(k, "").strip() for k in ("user", "action", "date")}
    query = AuditLog.query
    if filters["user"]:
        query = query.filter(AuditLog.username == filters["user"])
    if filters["action"]:
        query = query.filter(AuditLog.action == filters["action"])
    if filters["date"]:
        try:
            day = datetime.strptime(filters["date"], "%Y-%m-%d")
        except ValueError:
            flash("Date must be in YYYY-MM-DD form.")
        else:
            query = query.filter(AuditLog.timestamp >= day,
                                 AuditLog.timestamp < day + timedelta(days=1))
    # ponytail: latest 200, no paging; add paging when admins need older entries on screen.
    entries = query.order_by(AuditLog.id.desc()).limit(200).all()
    usernames = [u for (u,) in db.session.query(AuditLog.username).distinct() if u]
    actions = [a for (a,) in db.session.query(AuditLog.action).distinct()]
    return render_template("audit_log.html", entries=entries, filters=filters,
                           usernames=sorted(usernames), actions=sorted(actions))
