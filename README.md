# BANTAY

Barangay Anunas Incident Tracking and Analysis System. This is a Flask and SQLite web app for barangay blotter records. Sensitive text fields are encrypted with AES-256-GCM, access is role-based (admin, reviewer, encoder), and party names are hidden from every role except admin.

The thesis's Appendix L (User's Manual) has the full manual. This file covers installation and first start.

## Install

1. Install Python 3.12. Then install the packages:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   For the classifier, install Ollama and pull the deployed model and its check partner:
   `ollama pull gemma4:e4b` and `ollama pull aisingapore/Gemma-SEA-LION-v3-9B-IT:q4_k_m`.

2. Generate the two keys:

   ```
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   python -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"
   ```

   Save them in `instance/.env`. Git ignores this file.

   ```
   SECRET_KEY=<first value>
   BANTAY_ENC_KEY=<second value>
   BANTAY_OLLAMA=1
   ```

   The app refuses to start without either key. Keep a copy of `BANTAY_ENC_KEY` somewhere other than this computer. If you lose it, you lose every encrypted narrative.

## First start

```
python app.py
```

Then open http://localhost:5000.

- On the first start, the app creates three accounts: `admin`, `reviewer` and `encoder`. Each gets its own random one-time password. The passwords are printed once in the console window, like this:

  ```
  BANTAY first start - one-time passwords (shown only this once):
      admin      Xk3...
      reviewer   ...
      encoder    ...
  ```

  Copy them before you close the window. They are not stored anywhere else and are not shown again.
- At its first sign-in, each account goes straight to the **Change Password** screen. No other page opens until a new password is set.
  - The new password must have at least 10 characters.
  - It must differ from the current password.
  - It cannot be `changeme123`.
- Upgrading an older installation: start the app as usual. On start, it adds the new columns and tables to the existing `instance/bantay.db` without changing any record. Any account still using the old shared password `changeme123` must change it at its next sign-in.

Each user can later change their own password from **Change password** at the bottom of the sidebar.

### Debug mode

Debug mode is **off** by default. When it is off, the Werkzeug interactive debugger and the auto-reloader are off too. Turn it on only on a development machine:

```
set BANTAY_DEBUG=1        (PowerShell: $env:BANTAY_DEBUG="1")
python app.py
```

Never set `BANTAY_DEBUG=1` on the barangay computer. The debugger runs any Python code typed into the browser, and the app listens on the hall network (`0.0.0.0`). `FLASK_DEBUG` does not turn on BANTAY's debug mode. `flask run --debug` is a separate development command; do not use it.

### Web server

`python app.py` uses Flask's built-in server. No production server (waitress or gunicorn) is installed. The built-in server is enough for a few users on the hall network, but it is not built for public exposure. It also serves plain HTTP, so passwords travel unencrypted on the local network.

## Accounts the app has no screen for

The app has no screen for adding users or resetting a forgotten password. Do both from the command line in the BANTAY folder:

```
flask --app app shell
>>> import secrets
>>> from bantay import db
>>> from bantay.models import User
>>> pw = secrets.token_urlsafe(9); print(pw)            # one-time password to hand over
>>> u = User(username="kag.santos", role="encoder", must_change_password=True)   # new account
>>> # or, to reset: u = User.query.filter_by(username="encoder").first(); u.must_change_password = True
>>> u.set_password(pw)
>>> db.session.add(u); db.session.commit()
```

Because `must_change_password=True` is set, that person must choose their own password at their next sign-in.

## Audit log

The app records these actions in an `audit_log` table:

- sign-ins (successful and failed) and sign-outs
- password changes
- opening a record's page
- creating, importing, editing, reviewing and deleting records
- .docx exports
- any action refused because of the user's role

Each entry holds only the time (UTC), the account, its role, the action, the record ID, the outcome and the IP address. It never holds narrative text, names or any other record content. The app has no way to edit or delete entries, and the database rejects UPDATE and DELETE on the table. An admin can read the latest entries under **Admin → Audit Log**. The filters are user, action and date.

## Backups

Stop the app, then copy `instance/bantay.db` to another drive. A backup is useless without `BANTAY_ENC_KEY`, so keep the key backed up separately.

## Tests

```
python -m pytest -q
```
