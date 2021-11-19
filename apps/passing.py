# -*- coding: utf-8 -*-
import base64
import io
import os
from io import BytesIO

import dash_bootstrap_components as dbc
import dvwtools.read as dv
import dvwtools.stats as dvstats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
import plotly.express as px
import seaborn as sns
from PIL import Image
from dash import dash_table
from dash import dcc
from dash import html
from dash.dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output

from app import app


# Loads up the style for the page.
# No need to change it unless we want to change the graphics of the controls
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = [dbc.themes.BOOTSTRAP]


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"

operators = [
    ["ge ", ">="],
    ["le ", "<="],
    ["lt ", "<"],
    ["gt ", ">"],
    ["ne ", "!="],
    ["eq ", "="],
    ["contains "],
    ["datestartswith "],
]

# PASSING FUNCTIONS
def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find("{") + 1 : name_part.rfind("}")]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', "`"):
                    value = value_part[1:-1].replace("\\" + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


def do_table_filtering(dff, page_current, page_size, sort_by, filter_by):
    filtering_expressions = filter_by.split(" && ")

    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ("eq", "ne", "lt", "le", "gt", "ge"):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == "contains":
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == "datestartswith":
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col["column_id"] for col in sort_by],
            ascending=[col["direction"] == "asc" for col in sort_by],
            inplace=False,
        )

    return dff.iloc[page_current * page_size : (page_current + 1) * page_size].to_dict(
        "records"
    )


def discrete_background_color_bins(df_to_colorize, columns, inverse_colorscale, n_bins=5):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]

    df_numeric_columns = df_to_colorize[columns]

    df_max = df_numeric_columns.max()
    df_min = df_numeric_columns.min()

    mode = 'linear'
    if mode == 'linear':
        ranges = np.linspace(df_min, df_max, len(bounds))
    else:
        ranges = np.geomspace(df_min, df_max, len(bounds))


    # colorscale = ['RdYlGn_r' if reverse else 'RdYlGn' for reverse in inverse_colorscale]

    styles = []
    legend = []

    alpha = 0.5
    for column in range(len(df_numeric_columns.columns)):
        col_name = df_numeric_columns.columns[column]
        colscale = colorlover.scales[str(n_bins)]['div']['RdYlGn']
        if inverse_colorscale[column]:
            colscale = colscale[::-1]

        for i in range(1, len(bounds)):
            min_bound = ranges[i - 1, column]
            max_bound = ranges[i, column]

            backgroundColor = colscale[i - 1]
            # color = 'white' if i > len(bounds) / 2. else 'inherit'
            color = 'inherit'

            # print(f'col_name {col_name}, minbound {min_bound}, maxbound {max_bound}')
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=col_name, min_bound=min_bound, max_bound=max_bound),
                    'column_id': col_name
                },
                'backgroundColor': rgb_to_rgba(backgroundColor, alpha),
                'color': color
            })

    return styles


def colorize_passing_columns(data):
    styles = discrete_background_color_bins(data,
                                            columns=['good_perc', 'perfect_perc', 'errors_perc', 'efficiency'],
                                            inverse_colorscale=[False, False, True, False],
                                            n_bins=10)
    return styles


# INITIALIZATION
# Loads the list of passes from a csv file

df_all = pd.read_parquet(path=config.s3_bucket + 'complete_df.parquet.gzip',
                     storage_options={"key": config.access_key, "secret": config.secret_key})
teams_list = pd.read_parquet(path=config.s3_bucket + 'teams.parquet.gzip',
                             storage_options={"key": config.access_key, "secret": config.secret_key})
players_list = pd.read_parquet(path=config.s3_bucket + 'players.parquet.gzip',
                               storage_options={"key": config.access_key, "secret": config.secret_key})

df = dv.get_reception_df(df_all)
giocatori = df["CodiceGiocatore"].unique()
partite = df["Partita"].unique()
rotazioni = df["iz"].unique()
tipo_battute = df.type.unique()

# Loads the image of the volleyball court
path_image = os.path.join("assets", "mezzoCampo.png")

# Select different colors for different quality of passes.
# The RdBu palette goes from deep red to deep blue (diverging)
colors = [
    px.colors.diverging.RdBu[0],
    px.colors.diverging.RdBu[1],
    px.colors.diverging.RdBu[2],
    px.colors.diverging.RdBu[7],
    px.colors.diverging.RdBu[9],
    px.colors.diverging.RdBu[10],
]

# Just some handling of the volleyball court image
encoded_image = base64.b64encode(open(path_image, "rb").read())

encoded_image_up = base64.b64encode(
    open(os.path.join("assets", "mezzoCampo_up.png"), "rb").read()
)
traces = ["Perfect", "Positive", "Efficiency"]

giocatori_ddw = dcc.Dropdown(
    id="giocatori-column",
    options=[{"label": i, "value": i} for i in giocatori],
    value=giocatori,
    multi=True,
)

partite_ddw = dcc.Dropdown(
    id="partite-column",
    options=[{"label": i, "value": i} for i in partite],
    value=partite,
    multi=True,
)

tipologia_ddw = dcc.Dropdown(
    id="tipologia-column",
    options=[{"label": i, "value": i} for i in tipo_battute],
    value=tipo_battute,
    multi=True,
)

rotazioni_ddw = dcc.Dropdown(
    id="rotazioni-column",
    options=[{"label": i, "value": i} for i in rotazioni],
    value=rotazioni,
    multi=True,
)

partenza_sld = dcc.RangeSlider(
    id="partenza-slider",
    min=0,
    max=100,
    value=[0, 100],
    marks={0: "1", 25: "9", 50: "6", 75: "7", 100: "5"},
)

teams_ddw = dcc.Dropdown(
    id="teams-column",
    options=[
        {"label": name, "value": code}
        for name, code in zip(teams_list.SquadraNome, teams_list.index)
    ],
    value=teams_list.index[0],
    clearable=False,
)

grouping_players_rbn = dbc.RadioItems(
    id="group-players",
    options=[
        {"label": "By Team", "value": 0},
        {"label": "By Players", "value": 1},
    ],
    value=0,
)


grouping_time_rbn = dcc.Dropdown(
    id="group-time",
    options=[
        {"label": "Competition", "value": 0},
        {"label": "Round", "value": 1},
        {"label": "Game", "value": 2},
        # {'label': 'Each Set', 'value': 3},
        {"label": "Set (General)", "value": 4},
        {"label": "Difficulty", "value": 5},
    ],
    value=0,
    clearable=False,
)

xyplot_performance_ddw = dcc.Dropdown(
    id="xyplot_performance",
    options=[
        {"label": "By Efficiency", "value": 0},
        {"label": "By Occurrence", "value": 1},
    ],
    value=0,
    multi=False,
    clearable=False,
)

parametri_prestazione_ddw = dcc.Dropdown(
    id="prestazione-column",
    options=[{"label": i, "value": i} for i in traces],
    value="Positive",
    clearable=False,
)

# output = dcc.Graph(id='XvsY-Pass')

timeline_plot = dcc.Graph(style={"height": "600px"}, id="timeline-plot")

stats_lollipop = dcc.Graph(id="stats-lollipop-reception")

stats_winlose = dcc.Graph(style={"height": "600px"}, id="stats-winlose-bar-reception")

sankey_pass_plot = dcc.Graph(style={"height": "600px"}, id="sankey-pass")

pad = 4


# LAYOUT
def layout():
    layout_app = (
        html.Div(
            [
                dbc.Container(
                    [
                        dbc.Row(
                            [dbc.Col(html.H1("Passing performance"), className="mb-2")]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H6(
                                        children="Visualize passing stats for South Korea National Team"
                                    ),
                                    className="mb-4",
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="Select team",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                    lg=12,
                                    md=12,
                                    xs=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(teams_ddw),
                                    className="mt-4",
                                    lg=12,
                                    md=12,
                                    xs=12,
                                ),
                            ]
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="reception kde",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [html.Img(id="kde-reception")],
                                            style={"textAlign": "center"},
                                        )
                                    ],
                                    lg=12,
                                    md=12,
                                    xs=12,
                                )
                            ]
                        ),
                        # dbc.Row([
                        #     dbc.Col(
                        #         dbc.FormGroup([
                        #             dbc.Label("Plot data by:"),
                        #             xyplot_performance_ddw
                        #         ]), lg=2, md=4, xs=12, style={'padding-top': 80}),
                        #     dbc.Col(html.Div(output), lg=10, md=8, xs=12),
                        # ]),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="Timeline plot",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                )
                            ]
                        ),
                        html.Div(
                            [
                                dbc.Col(
                                    html.Label(
                                        "Group horizontal data by:",
                                        className="float-right",
                                    ),
                                    className="mt-4",
                                    lg=4,
                                    md=5,
                                    xs=12,
                                ),
                                dbc.Col(grouping_time_rbn, className="mt-4"),
                            ],
                            className="row g-3 align-items-center",
                        ),
                        dbc.Row([dbc.Col([timeline_plot], lg=12, md=14, xs=12)]),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="Team comparison",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dash_table.DataTable(
                                        id="table-reception-team",
                                        columns=[
                                            dict(name="Team", id="SquadraName"),
                                            dict(name="Total", id="total"),
                                            dict(
                                                name="Positive (%)",
                                                id="good_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Perfect (%)",
                                                id="perfect_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Errors (%)",
                                                id="errors_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Efficiency (%)",
                                                id="efficiency",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                        ],
                                        page_current=0,
                                        page_size=20,
                                        page_action="custom",
                                        filter_action="custom",
                                        filter_query="{total} > 10",
                                        sort_action="custom",
                                        sort_mode="multi",
                                        sort_by=[
                                            {
                                                "column_id": "good_perc",
                                                "direction": "desc",
                                            }
                                        ],
                                        # style_as_list_view=True,
                                        style_cell={
                                            "padding": "5px",
                                            "font-family": "Lucida Sans Typewriter",
                                            "fontSize": 14,
                                        },
                                        style_header={
                                            "backgroundColor": "white",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    lg=12,
                                    md=12,
                                    xs=12,
                                    style={"padding": 35},
                                ),
                            ],
                            style={"padding": pad},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        id="table-reception-team-container",
                                    ),
                                    lg=12,
                                    md=12,
                                    xs=12,
                                )
                            ],
                            style={"padding-top": 0},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="Player comparison",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.Div(stats_lollipop), lg=12, md=12, xs=12),
                            ],
                            className="mt-4",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dash_table.DataTable(
                                        id="table-reception",
                                        columns=[
                                            dict(name="Player", id="GiocatoreName"),
                                            dict(name="Total", id="total"),
                                            dict(
                                                name="Positive (%)",
                                                id="good_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Perfect (%)",
                                                id="perfect_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Errors (%)",
                                                id="errors_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Efficiency (%)",
                                                id="efficiency",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                        ],
                                        page_current=0,
                                        page_size=20,
                                        page_action="custom",
                                        filter_action="custom",
                                        filter_query="{total} > 10",
                                        sort_action="custom",
                                        sort_mode="multi",
                                        sort_by=[
                                            {
                                                "column_id": "good_perc",
                                                "direction": "desc",
                                            }
                                        ],
                                        # style_as_list_view=True,
                                        style_cell={
                                            "padding": "5px",
                                            "font-family": "Lucida Sans Typewriter",
                                            "fontSize": 14,
                                        },
                                        style_header={
                                            "backgroundColor": "white",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    lg=12,
                                    md=12,
                                    xs=12,
                                    style={"padding": 35},
                                ),
                            ],
                            style={"padding": pad},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        id="table-reception-container",
                                    ),
                                    lg=12,
                                    md=12,
                                    xs=12,
                                )
                            ],
                            style={"padding-top": 0},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="Sets won/lost, by passing performance",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                )
                            ],
                            style={"padding-top": 60},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.Div(), lg=2, md=4, xs=12),
                                dbc.Col(
                                    html.Div(stats_winlose),
                                    lg=6,
                                    md=8,
                                    xs=12,
                                    style={"padding": pad},
                                ),
                            ],
                            style={"padding-top": 0},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="Passing stats based on opponent serving player",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                    lg=12,
                                    md=12,
                                    xs=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dash_table.DataTable(
                                        id="table-paging-with-graph",
                                        columns=[
                                            dict(
                                                name="Serve  Player",
                                                id="ServeGiocatoreName",
                                            ),
                                            dict(name="Opp. Team", id="TeamAvv"),
                                            dict(name="Total", id="total"),
                                            dict(
                                                name="Positive (%)",
                                                id="good_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Perfect (%)",
                                                id="perfect_perc",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Efficiency (%)",
                                                id="efficiency",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                        ],
                                        # columns=[
                                        #     {"name": i, "id": i} for i in sorted(bestServers.columns)
                                        # ],
                                        page_current=0,
                                        page_size=20,
                                        page_action="custom",
                                        filter_action="custom",
                                        filter_query="{total} > 10",
                                        sort_action="custom",
                                        sort_mode="multi",
                                        sort_by=[
                                            {
                                                "column_id": "efficiency",
                                                "direction": "asc",
                                            }
                                        ],
                                        # style_as_list_view=True,
                                        style_cell={
                                            "padding": "5px",
                                            "font-family": "Lucida Sans Typewriter",
                                            "fontSize": 14,
                                        },
                                        style_header={
                                            "backgroundColor": "white",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    lg=12,
                                    md=12,
                                    xs=12,
                                    style={"padding": 35},
                                ),
                            ],
                            style={"padding": pad},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        id="table-paging-with-graph-container",
                                    ),
                                    lg=12,
                                    md=12,
                                    xs=12,
                                )
                            ],
                            style={"padding-top": 0},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        html.H3(
                                            children="How pass transforms into points",
                                            className="text-center text-light bg-dark",
                                        ),
                                        body=True,
                                        color="dark",
                                    ),
                                    className="mt-4",
                                    lg=12,
                                    md=12,
                                    xs=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                # dbc.Col(html.Div(), lg=2, md=4, xs=12, style={'padding-top': 20}),
                                dbc.Col(
                                    html.Div(sankey_pass_plot), lg=12, md=12, xs=12
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    )
    return layout_app


# # CALLBACK FUNCTIONS
@app.callback(
    Output("stats-lollipop-reception", "figure"), [Input("teams-column", "value")]
)
def update_lollipop_attack(team_code):
    fund = dvstats.Fundamental.reception
    list_players = players_list.loc[players_list.SquadraCodice == team_code]
    fig = dvstats.update_lollipop_team_players(
        df[df.SquadraCodice == team_code],
        dvstats.plots_viz_dict[fund]["performance"],
        fund,
        list_players,
        dvstats.plots_viz_dict[fund]["performance"][0],
    )
    return fig


@app.callback(
    Output("stats-winlose-bar-reception", "figure"), [Input("teams-column", "value")]
)
def update_bin_plot(selected_team_code):
    fund = dvstats.Fundamental.reception
    return dvstats.get_binned_winning_stats_plot(
        df[df.SquadraCodice == selected_team_code], fund, "good_perc", 11
    )


@app.callback(
    Output("timeline-plot", "figure"),
    [Input("group-time", "value"), Input("teams-column", "value")],
)
def update_timeline_graph(radio_index, selected_team_code):
    all_teams = dvstats.get_teams_list(df)
    teams_df, selected_field_name, mode = dvstats.get_team_timeline_stats(
        radio_index, all_teams, df, dvstats.Fundamental.reception
    )
    return dvstats.get_timeline_plot_team(
        teams_df,
        selected_team_code,
        selected_field_name,
        mode,
        dvstats.Fundamental.reception,
        ["good_perc", "perfect_perc"],
    )


@app.callback(Output("sankey-pass", "figure"), [Input("teams-column", "value")])
def update_sankey(team_code):
    nodes, links = dvstats.get_sankey_data(df, "valutazione", "SideoutAttackGrade")
    return dvstats.get_sankey_plot(nodes, links, team_code)


@app.callback(Output("kde-reception", "src"), [Input("teams-column", "value")])
def update_kde(team_code):
    data_df = df[df.SquadraCodice == team_code]
    img = Image.open(io.BytesIO(base64.b64decode(encoded_image_up)))
    fig, axes = plt.subplots(figsize=(6, 6))
    sns.kdeplot(
        x=data_df.x_a,
        y=data_df.y_a,
        ax=axes,
        fill=True,
        alpha=0.9,
        levels=30,
        cmap="OrRd",
    )

    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    plt.xlim(0, 100)
    plt.ylim(50, 100)
    plt.imshow(img, extent=[0, 100, 50, 100], aspect="auto")

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")

    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
    return "data:image/png;base64,{}".format(data)


@app.callback(
    Output("table-paging-with-graph", "data"),
    [
        Input("table-paging-with-graph", "page_current"),
        Input("table-paging-with-graph", "page_size"),
        Input("table-paging-with-graph", "sort_by"),
        Input("table-paging-with-graph", "filter_query"),
        Input("teams-column", "value"),
    ],
)
def update_table_serve(page_current, page_size, sort_by, filter_by, code):
    dff = dvstats.get_server_table(df, code)
    return do_table_filtering(dff, page_current, page_size, sort_by, filter_by)


@app.callback(
    [Output('table-reception', "data"),
     Output('table-reception', "style_data_conditional")],
    [Input('table-reception', "page_current"),
     Input('table-reception', "page_size"),
     Input('table-reception', "sort_by"),
     Input('table-reception', "filter_query"),
     Input('teams-column', 'value')])
def update_table(page_current, page_size, sort_by, filter_by, code):
    dff = dvstats.get_stats_table(df[df.SquadraCodice == code], dvstats.Fundamental.reception, dvstats.Granularity.players, players_list)
    return do_table_filtering(dff, page_current, page_size, sort_by, filter_by), colorize_passing_columns(dff)


@app.callback(
    [Output('table-reception-team', "data"),
     Output('table-reception-team', "style_data_conditional")],
    [Input('table-reception-team', "page_current"),
     Input('table-reception-team', "page_size"),
     Input('table-reception-team', "sort_by"),
     Input('table-reception-team', "filter_query"),
     Input('teams-column', 'value')])
def update_table(page_current, page_size, sort_by, filter_by, code):
    dff = dvstats.get_stats_table(df, dvstats.Fundamental.reception, dvstats.Granularity.teams, teams_list)
    return do_table_filtering(dff, page_current, page_size, sort_by, filter_by), colorize_passing_columns(dff)
