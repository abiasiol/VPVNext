import dash
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.LUX]
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
)
app.title = "VPV Next"
server = app.server
