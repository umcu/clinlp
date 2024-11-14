"""Entrypoint for the Information Extraction demo app."""

import dash_bootstrap_components as dbc
from dash import Dash

from clinlp_apps.ie_demo.src.view import layout

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.layout = layout

if __name__ == "__main__":
    app.run_server(debug=True)
