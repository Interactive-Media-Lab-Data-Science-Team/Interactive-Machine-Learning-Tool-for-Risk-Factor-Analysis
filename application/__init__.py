"""Initialize Flask app."""
from flask import Flask
from flask_assets import Environment


def create_app():
    """Construct core Flask application with embedded Dash app."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')
    app.config["SQLALCHEMY_DATABASE_URI"] = "postgres://gnxvbjngizutpx:37fa76dfe3c3469ad91e9f6e1c294b2f2257f42d1111d5c8080861bb1c013915@ec2-34-237-236-32.compute-1.amazonaws.com:5432/dvnem1a7777qs"
    assets = Environment()
    assets.init_app(app)

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes
        from .assets import compile_static_assets

        # Import Dash application
        from .plotlydash.dataDownload import dataDownload
        # from .plotlydash.testLoad import test1
        # from .plotlydash.EDA import create_EDA
        # from .plotlydash.riskFactorAnalysis import create_RFA
        # from .plotlydash.PCA import create_PCA
        # from .plotlydash.page3 import create_page3
        # from .plotlydash.page4 import create_page4
        # tryout(app)
        dataDownload(app)

        # create_PCA(app)

        # Compile static assets
        compile_static_assets(assets)

        return app
