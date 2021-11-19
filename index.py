from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import config
from app import app, server
from apps import passing, attack



# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Passing", href="/passing"),
        dbc.DropdownMenuItem("Attack", href="/attack"),
    ],
    nav=True,
    in_navbar=True,
    label="Explore",
)


def get_dataset_stats(data_df):
    return {
        "rows": len(data_df),
        "attacks": len(data_df[data_df.fundamental == "A"]),
        "sets": len(data_df[data_df.fundamental == "E"]),
        "receptions": len(data_df[data_df.fundamental == "R"]),
        "defenses": len(data_df[data_df.fundamental == "D"]),
        "blocks": len(data_df[data_df.fundamental == "B"]),
        "serves": len(data_df[data_df.fundamental == "S"]),
        "freeballs": len(data_df[data_df.fundamental == "F"]),
        "teams": len(data_df.SquadraCodice.unique()),
        "competitions": len(data_df.Competizione.unique()),
    }


stats = get_dataset_stats(
    pd.read_parquet(path=config.s3_bucket + 'complete_df.parquet.gzip',
                    storage_options={"key": config.access_key, "secret": config.secret_key})
)

card_passing = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.CardImg(
                        src=app.get_asset_url("kde4.png"),
                        className="img-fluid",
                    ),
                    className="col-md-4",
                ),
                dbc.Col(
                    dbc.CardBody(
                        [
                            html.H2("Pass", className="card-title"),
                            html.P(
                                "Reception statistics and visualizations. "
                                "Go to the pass page.",
                                className="card-text",
                            ),
                            # html.Small(
                            #     "Last updated 3 mins ago",
                            #     className="card-text text-muted",
                            # ),
                        ]
                    ),
                    className="col-md-8",
                ),
            ],
            className="g-0 d-flex align-items-center",
        ),
        dcc.Link("", href="/passing", className="stretched-link"),
    ],
    className="mb-3 border border-4 border-secondary rounded",
    style={"maxWidth": "800px"},
)

card_attack = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.CardImg(
                        src=app.get_asset_url("korea-wonlost2.png"),
                        className="img-fluid",
                    ),
                    className="col-md-4",
                ),
                dbc.Col(
                    dbc.CardBody(
                        [
                            html.H2("Attack", className="card-title"),
                            html.P(
                                "Attack statistics and visualizations. Go to the attack page.",
                                className="card-text",
                            ),
                            # html.Small(
                            #     "Last updated 3 mins ago",
                            #     className="card-text text-muted",
                            # ),
                        ]
                    ),
                    className="col-md-8",
                ),
            ],
            className="g-0 d-flex align-items-center",
        ),
        dcc.Link("", href="/attack", className="stretched-link"),
    ],
    className="mb-3 border border-4 border-secondary rounded",
    style={"maxWidth": "800px"},
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/kr.svg", height="30px")),
                        dbc.Col(dbc.NavbarBrand("VPV Next", className="ml-2")),
                    ],
                    align="center",
                    # no_gutters=True,
                ),
                href="/",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)


def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        html.Div(id="page-content", className="container-fluid"),
    ],
    className="container-fluid",
)


index_page = html.Div(
    [
        dbc.Container(
            [
                html.H1("South Korea National Team Database 2021"),
                html.Br(),
                html.H6(
                    "Welcome to the South Korea National Team Database, curated by Andrea Biasioli."
                ),
                html.Br(),
                html.Div(
                    f'This dataset currently contains {stats["rows"]:,} data points:'
                ),
                html.Ul(
                    [
                        html.Li(f'{stats["serves"]:,} serves'),
                        html.Li(f'{stats["receptions"]:,} receptions'),
                        html.Li(f'{stats["sets"]:,} sets'),
                        html.Li(f'{stats["attacks"]:,} attacks'),
                        html.Li(f'{stats["blocks"]:,} blocks'),
                        html.Li(f'{stats["defenses"]:,} defenses'),
                        html.Li(f'{stats["freeballs"]:,} freeballs'),
                    ]
                ),
                html.Div(
                    f'The data was collected in '
                    f'{stats["competitions"]:,} different competions, '
                    f'and {stats["teams"]:,} teams are represented.'
                ),
                html.Br(),
                html.P(
                    "Please use the navigation bar on "
                    "the top-right side of the page or the "
                    "following links to visually explore the data."
                ),
                html.Br(),
                card_passing,
                card_attack,
            ]
        ),
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/passing":
        return passing.layout()
    elif pathname == "/attack":
        return attack.layout_attack()

    return index_page


if __name__ == "__main__":
    app.run_server(port=7000, debug=True)
