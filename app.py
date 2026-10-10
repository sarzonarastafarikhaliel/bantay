from bantay import create_app
from bantay.seed import seed_default_users

app = create_app()

with app.app_context():
    seed_default_users()

if __name__ == "__main__":
    # app.debug comes from BANTAY_DEBUG (see create_app): off unless set to 1.
    # With it off, Flask's run() also leaves the Werkzeug debugger and the
    # reloader off - the debugger runs arbitrary Python for anyone who can reach
    # the port, and host 0.0.0.0 makes that the whole barangay hall LAN.
    app.run(host="0.0.0.0", debug=app.debug)
