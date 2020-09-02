"""Initialize Flask app."""
from flask import Flask
from flask_assets import Environment


def create_app():
    """Construct core Flask application with embedded Dash app."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')
    assets = Environment()
    assets.init_app(app)

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes
        from .assets import compile_static_assets

        # Import Dash application
        from .plotlydash.dataDownload import dataDownload
        # from .plotlydash.testLoad import test1
        # from .plotlydash.page2 import create_page2
        # from .plotlydash.page3 import create_page3
        # from .plotlydash.page4 import create_page4
        dataDownload(app)

        # Compile static assets
        compile_static_assets(assets)

        return app
