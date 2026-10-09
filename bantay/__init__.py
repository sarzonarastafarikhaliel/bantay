import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"


def create_app(config_object="config.Config", overrides=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)
    if overrides:
        app.config.update(overrides)
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from .routes.auth import auth_bp
    from .routes.records import records_bp
    from .routes.review import review_bp
    from .routes.dashboard import dashboard_bp
    from .routes.analytics import analytics_bp
    from .routes.scan import scan_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(records_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(scan_bp)

    from .ml.infer import Classifier
    app.classifier = Classifier(app.config["MODEL_DIR"])

    from .narrative import redact_names
    from .normalize import place_names

    app.jinja_env.globals["place_names"] = place_names   # Lugar <datalist> suggestions

    @app.template_filter("visible_narrative")
    def visible_narrative(text):
        """Party names in the narrative are visible only to admin - every
        other role sees the COMPLAINANT/RESPONDENT lines masked. Applied at
        render time in templates, not at query time, so search and editing
        (which need the real text) are unaffected."""
        if current_user.is_authenticated and current_user.role == "admin":
            return text or ""
        return redact_names(text)

    @app.template_filter("confidence_label")
    def confidence_label(value):
        """model_confidence is 0.8/0.5/0.3 from whether two models agree
        (routes/records.py), not a probability - show the category, not a %."""
        if value >= 0.6:
            return "Models agree"
        if value >= 0.4:
            return "Unverified (check model unavailable)"
        return "Models disagree"

    with app.app_context():
        db.create_all()
        from .routes.records import sync_review_queue
        sync_review_queue()

    return app
