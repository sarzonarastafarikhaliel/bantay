# R7 — Security defaults: summary of changes

Branch: `r7-security-hardening`. The thesis .docx was not edited. The suggested wording for it is in section 5 below.

## 1. Before and after

| Problem found in review | Before | After |
|---|---|---|
| Shared default password | The first start created `admin`, `reviewer` and `encoder`, all with the password `changeme123`. | Each seeded account gets its own random 12-character one-time password (`secrets.token_urlsafe`). It is printed once to the server console and stored only as a hash. No password is hard-coded. |
| No forced change | Nothing made anyone change the default. | New `user.must_change_password` column. It is set on seeded accounts, and on every start it is set on any account whose hash still matches `changeme123`. A `before_request` hook redirects every route except the change-password page, logout and static files, so typing a URL does not get around it. Existing databases get the column automatically on start, with no data loss. |
| No password-change screen | Passwords could be changed only from `flask shell`. | `/change-password` asks for the current password, the new password and a confirmation. The new password must have at least 10 characters and cannot equal the current password or `changeme123`. Hashing is the same as before (Werkzeug default, `scrypt:32768:8:1`). The form has a per-session CSRF token. A successful change clears the flag and re-issues the session. Every user reaches it from the sidebar. |
| Debug mode on | `app.run(debug=True, host="0.0.0.0")` exposed the Werkzeug debugger (remote code execution) to the LAN. | `create_app` sets `DEBUG` only when `BANTAY_DEBUG=1`, and ignores `FLASK_DEBUG`. `app.py` passes `app.debug` through, so the debugger and the reloader are both off by default. There is no Dockerfile or start script; `app.py` is the only entry point. |
| No audit log | No record of who viewed, edited, exported or deleted a record. | New `audit_log` table with these columns: id, timestamp (UTC), user_id, username, role, action, target_type, target_id, outcome, ip. Rows are written by one helper, `bantay.audit.audit()`, which accepts IDs and action names only. The log is append-only: no route edits it, and SQLite triggers abort UPDATE and DELETE. `/audit` is an admin-only, read-only page (latest 200 entries; filters for user, action and date). Non-admins get a 403. |

**What the audit log records:**

| Event | `action` value |
|---|---|
| Sign-in, success or failure | `login` |
| Sign-out | `logout` |
| Password change, success or failure | `password_change` |
| Viewing a record's detail page | `record.view` |
| Creating a record | `record.create` |
| Importing records | `record.import` |
| Editing a record | `record.edit` |
| Resolving a record in the Review Queue | `record.review` |
| Sending a record back to review | `record.send_to_review` |
| Deleting one record | `record.delete` |
| Deleting all records | `record.delete_all` |
| Exporting one record | `record.export` |
| Exporting the filtered list | `record.export_bulk` |
| Any role-restricted action refused | the endpoint name, with outcome `denied` |

For a failed sign-in, the attempted name is recorded only when it is a real account. A mistyped password or a resident's name typed into the username box never reaches the log. For a filtered export, the filter values (which can include free-text search) are not logged.

### Choices made

- **Random one-time passwords instead of a fixed default.** This breaks no existing test: no test used the seeded accounts, because every test creates its own users. The only documented setup that relied on `changeme123` was Appendix L, and section 5 below rewrites it. The cost: if nobody copies the console output at first start, the passwords are gone, and an admin must reset them from `flask shell` (the README shows how).
- **CSRF.** The app had no CSRF protection anywhere, and Flask-WTF is not installed. The change-password form now has its own per-session token. It uses only the standard library (`secrets` and `hmac.compare_digest`) and refuses a missing token. Every other form is still without a token. As an app-wide one-line measure, the session cookie is now `SameSite=Lax`, so browsers withhold it on cross-site POSTs.
- **Session rotation.** On a successful password change the session is cleared and the user is signed in again, so the browser receives a newly issued cookie and a new CSRF token. Flask sessions are signed client-side cookies, so a copy of the old cookie stays valid. Invalidating it would need server-side sessions or a per-user session version. This was not implemented.
- **Production server.** Neither waitress nor gunicorn is a dependency, so the documented start command is still `python app.py` (the Flask built-in server, now with debug off). The README says so.

## 2. Tests

| | Count |
|---|---|
| Full suite before R7 (working tree) | **268 passed** |
| Full suite after R7 | **287 passed**, 0 failed (218 warnings, up from 70: mostly the same `datetime.utcnow()` deprecation notice, now also raised by the audit-log timestamp default) |
| New tests (`tests/test_r7_security.py`) | 19 |
| Pre-existing tests changed | none |

The 268 includes the 9 tests in `tests/test_pool_188.py`, which is still untracked in git. Counting only committed test files, the numbers are 259 before and 278 after. The thesis's "256" came from an earlier run.

The new tests cover:

- **Debug mode:** `create_app` with and without `BANTAY_DEBUG`, and with `FLASK_DEBUG=1` alone. They also run `app.py` as `__main__` in a subprocess, with the server call stubbed, and check what it would pass for `use_debugger` and `use_reloader`.
- **Seeded accounts:** random, distinct, flagged one-time passwords, printed once.
- **Forced change:** a flagged user is redirected from every registered route and method (25 combinations, taken from `app.url_map`), and none of those views runs.
- **Change-password rules:** too short, same as current, wrong current password, mismatched confirmation, `changeme123`, missing or forged CSRF token, and a success that clears the flag, re-issues the session and keeps the same hash scheme.
- **Upgrade of an existing database:** a pre-R7 `user` table is upgraded in place, and re-running the upgrade changes nothing.
- **Audit log:**
  - one entry per listed action, in order, with the correct user, role, target and outcome;
  - a seeded distinctive narrative and name never appear in any `audit_log` row (checked by raw SQL);
  - non-admins get a 403 on the audit page;
  - the filters work;
  - UPDATE and DELETE are rejected.

A mutation check confirmed the tests catch the original bugs. With `debug=True` restored and the redirect hook disabled, 3 of them failed.

## 3. Files changed

- `app.py` and `bantay/__init__.py`: debug switch, redirect hook, schema upgrade on start.
- `bantay/models.py`: `User.must_change_password`, `AuditLog`.
- `bantay/seed.py`: one-time passwords, `upgrade_schema()`; removed the unused `seed_admin()`.
- `bantay/audit.py` (new): the single audit helper.
- `bantay/rbac.py`, `bantay/routes/{auth,records,review,dashboard}.py`: audit calls, change-password view, audit page.
- `bantay/templates/{change_password,audit_log}.html` (new) and `base.html`: sidebar links.
- `config.py` (one line: `SESSION_COOKIE_SAMESITE = "Lax"`) and `requirements.txt` (one line: `cryptography==50.0.0`, which was imported but not listed). Both are flagged extras.
- `tests/test_r7_security.py`, `README.md` (new), `.gitignore`.

RBAC decisions, encryption and name redaction are unchanged. The audit calls are added after each existing check or commit, and every pre-existing test passes unmodified. The classification pipeline was not touched.

## 4. Other security issues noticed (reported, not fixed)

1. **No CSRF tokens** on any form except change-password: new record, edit, delete, delete-all, send-to-review, import, review, check-review, reload-model, and the JSON `/records/predict`. `SameSite=Lax` is partial cover only. Logout is a GET request.
2. **No lockout or rate limit** on failed sign-ins. Failures are now logged, but nothing slows a password-guessing attempt.
3. **Session lifetime.** There is no idle timeout and no server-side revocation, because Flask sessions are signed client-side cookies. A copied cookie stays valid until `SECRET_KEY` changes.
4. **Plain HTTP on `0.0.0.0`.** Passwords and session cookies cross the hall LAN unencrypted, and `SESSION_COOKIE_SECURE` is off (it has to be, without HTTPS).
5. **Key custody.** `BANTAY_ENC_KEY` and `SECRET_KEY` sit in `instance/.env` in the same folder as `bantay.db`. This is already stated in §4.1.4.2.
6. **Open redirect via Referer.** `role_required` and `delete_record` redirect to `request.referrer`.
7. **No user-management screens.** Adding users and resetting passwords still requires `flask shell`.
8. **Audit-log integrity is application-level.** Anyone with file access can drop the triggers or edit the `.db` file. The log is not hash-chained or stored off-machine.
9. **Development server.** There is no production WSGI server (see above).

## 5. Suggested thesis wording

Each passage below states only what is now implemented.

### §4.1.4.2 Security Controls as Implemented

Replace the "Accounts and passwords" paragraph with:

> Accounts and passwords. On first start the system creates one account per role (admin, reviewer and encoder), each with its own randomly generated one-time password, which is printed once to the server console and not stored anywhere else. Each of these accounts, and any existing account whose password is still the former shared default, must change its password at its next sign-in: until it does, every page except the password-change screen and sign-out redirects to that screen. The password-change screen requires the current password, a new password of at least 10 characters that differs from the current one and from the former default, and a matching confirmation; it carries a per-session anti-forgery token, and a successful change clears the requirement and re-issues the session. Passwords are stored as salted scrypt hashes. There is no screen for adding a user or resetting a forgotten password, so the administrator does both from the command line (Appendix L, Section L.2). There is no lockout after repeated failed sign-ins, and the anti-forgery token protects the password-change form only.

Replace the "Logging and backups" paragraph with:

> Logging and backups. The system keeps an audit log of sign-ins, including failed attempts, sign-outs, password changes, the opening of a record's page, the creation, import, editing, review and deletion of records, .docx exports, and actions refused because of the user's role. Each entry stores the time in UTC, the account and its role, the action, the record identifier where there is one, the outcome, and the client IP address. It does not store narrative text, names or any other record content, which an automated test checks. The system offers no way to edit or delete entries, and the database rejects such changes, but a person with direct access to the database file could still alter it, so the log is append-only within the application rather than tamper-proof. Only the administrator can view it, on a read-only page filtered by user, action and date. There is no automatic backup: the database is a single SQLite file that the barangay must copy on a schedule.

Add after "Key custody":

> Server configuration. The system runs on Flask's built-in web server over plain HTTP on the barangay hall's local network. Debug mode, which would expose an interactive code console to that network, is off unless it is switched on explicitly for development.

### §4.1.4.1, abstract and Chapter 5

The test count is restated as **287** (abstract, §4.1.4.1 "Result: 287 passed, 0 failed", and Chapter 5's Objective 4 paragraph). In §4.1.4.1 and §3.3.4.4, add to the list of what the suite covers: "debug-mode defaults, the forced password change and its rules, the upgrade of an existing database, and the audit log, including a check that no narrative or name text is written to it." If `tests/test_pool_188.py` stays out of the repository, the count is 278.

Note on §1 (Scope and Limitations, "The system's data security measures are limited to…"): if you keep that sentence, add "an application-level audit log and a forced change of initial passwords" to the list of measures.

### Appendix L, Section L.2

Replace step 4 with:

> 4. Start the system with `python app.py` and open http://localhost:5000 in a browser. On first start the system creates three accounts named admin, reviewer and encoder and prints a one-time password for each in the console window. Copy them before closing the window; they are shown only once. Debug mode is off; do not set BANTAY_DEBUG=1 on the barangay computer.

Replace step 5 with:

> 5. Sign in with each account. The system opens the Change Password screen and no other page until a new password of at least 10 characters is set. To add a user or reset a forgotten password, the administrator runs the commands below in the BANTAY folder; the person must then choose their own password at their next sign-in.
>
> ```
> flask --app app shell
> >>> import secrets
> >>> from bantay import db
> >>> from bantay.models import User
> >>> pw = secrets.token_urlsafe(9); print(pw)
> >>> u = User(username="kag.santos", role="encoder", must_change_password=True)
> >>> u.set_password(pw); db.session.add(u); db.session.commit()
> ```

**L.3 Signing In and Out:** add "To change your password, click Change password at the bottom of the left sidebar."

**Table L1:** add a row "View the audit log: Administrator only".
