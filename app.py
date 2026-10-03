from bantay import create_app
from bantay.seed import seed_default_users

app = create_app()

with app.app_context():
    seed_default_users()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
