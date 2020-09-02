"""Core Flask app routes."""
from flask import render_template
from flask import current_app as app


@app.route('/')
def home():
    """Landing page."""
    return render_template(
        'index.jinja2',
        title='Interactive Machine Learning Tool',
        subtitle='Risk Factor Analysis for Medical Data',
        description='Embed Plotly Dash into your Flask applications.',
        template='home-template',
        body="This is a homepage served with Flask."
    )