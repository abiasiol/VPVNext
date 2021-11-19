# -*- coding: utf-8 -*-
import json
from typing import Set

import os
import config
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import base64
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dvwtools.read as dv
import colorlover
import dvwtools.stats as dvstats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dash import dash_table
from dash.dash_table.Format import Format, Scheme
from enum import Enum
from typing import NamedTuple
from dataclasses import dataclass

from app import app

# Loads up the style for the page. No need to change it unless we want to change the graphics of the controls
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = [dbc.themes.BOOTSTRAP]
# dash.Dash(__name__, external_stylesheets=external_stylesheets)
# pd.set_option('max_columns', None)

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


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def discrete_background_color_bins(df_to_colorize, columns, inverse_colorscale, n_bins=5):


    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]

    df_numeric_columns = df_to_colorize[columns]

    df_max = df_numeric_columns.max()
    df_min = df_numeric_columns.min()

    mode = "linear"
    if mode == "linear":
        ranges = np.linspace(df_min, df_max, len(bounds))
    else:
        ranges = np.geomspace(df_min, df_max, len(bounds))

    # colorscale = ['RdYlGn_r' if reverse else 'RdYlGn' for reverse in inverse_colorscale]

    styles = []
    legend = []

    alpha = 0.5
    for column in range(len(df_numeric_columns.columns)):
        col_name = df_numeric_columns.columns[column]
        colscale = colorlover.scales[str(n_bins)]["div"]["RdYlGn"]
        if inverse_colorscale[column]:
            colscale = colscale[::-1]

        for i in range(1, len(bounds)):
            min_bound = ranges[i - 1, column]
            max_bound = ranges[i, column]

            backgroundColor = colscale[i - 1]
            # color = 'white' if i > len(bounds) / 2. else 'inherit'
            color = "inherit"

            # print(f'col_name {col_name}, minbound {min_bound}, maxbound {max_bound}')
            styles.append(
                {
                    "if": {
                        "filter_query": (
                            "{{{column}}} >= {min_bound}"
                            + (
                                " && {{{column}}} < {max_bound}"
                                if (i < len(bounds) - 1)
                                else ""
                            )
                        ).format(
                            column=col_name, min_bound=min_bound, max_bound=max_bound
                        ),
                        "column_id": col_name,
                    },
                    "backgroundColor": rgb_to_rgba(backgroundColor, alpha),
                    "color": color,
                }
            )

    return styles


def colorize_attack_columns(data):
    styles = discrete_background_color_bins(
        data,
        columns=["kill", "blocked_perc", "errors_perc", "efficiency"],
        inverse_colorscale=[False, True, True, False],
        n_bins=10,
    )
    return styles


def get_sorted_players_list(p_list, data):
    count_df = data["CodiceGiocatore"].value_counts().to_frame("count")
    new_list = p_list.merge(count_df, how="left", left_index=True, right_index=True)
    new_list = new_list.sort_values(by="count", ascending=False)
    return new_list


def get_filtered_dataframe_attack(
    giocatore_value,
    partite_column,
    rotazioni_column,
    tipologia_column,
    ruoli_column,
    set_by_column,
    fondamentale_precedente,
):
    # print(fondamentale_precedente)
    filtro = (
        df["CodiceGiocatore"].isin(pd.Series(giocatore_value, dtype="string"))
        & df["Partita"].isin(pd.Series(partite_column, dtype="string"))
        & df["type"].isin(pd.Series(tipologia_column, dtype="string"))
        & df["iz"].isin(pd.Series(rotazioni_column, dtype="int"))
        & df["CurrentPlayerPosition"].isin(pd.Series(ruoli_column, dtype="string"))
        & df["SecondTouchGiocatoreName"].isin(pd.Series(set_by_column, dtype="string"))
        & df["FirstTouchFundamental"].isin(
            pd.Series(fondamentale_precedente, dtype="string")
        )
    )

    dff = df[filtro]
    return dff


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


def do_table_filtering(dff, page_current, page_size, sort_by, filter):
    filtering_expressions = filter.split(" && ")

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


# INITIALIZATION
# Loads the list of passes from a csv file
df_all = pd.read_parquet(path=config.s3_bucket + 'complete_df.parquet.gzip',
                     storage_options={"key": config.access_key, "secret": config.secret_key})

df = dv.get_attack_df(df_all)

teams_list = pd.read_parquet(path=config.s3_bucket + 'teams.parquet.gzip',
                             storage_options={"key": config.access_key, "secret": config.secret_key})
players_list = pd.read_parquet(path=config.s3_bucket + 'players.parquet.gzip',
                               storage_options={"key": config.access_key, "secret": config.secret_key})
rosters_df = pd.read_parquet(path=config.s3_bucket + 'rosters.parquet',
                             storage_options={"key": config.access_key, "secret": config.secret_key})


ruoli = df["CurrentPlayerPosition"].unique()
partite = df["Partita"].unique()
rotazioni = df["iz"].unique()

tipo_attacchi = df.type.unique()
removeAttackOthers = np.array("O")
tipo_attacchi_filtrato = np.setdiff1d(tipo_attacchi, removeAttackOthers)

set_by = df["SecondTouchGiocatoreName"].value_counts().reset_index()["index"]

setters_code_list = df["SetCode"].unique()
setters_list = players_list.loc[setters_code_list]["GiocatoreName"]

sorted_players_list = get_sorted_players_list(players_list, df)

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
traces = ["Kill", "Efficiency", "Quantity"]

attack_after_testo = ["Pass", "Defense", "Freeball", "Other"]
attack_after_codice = ["R", "D", "F", "O"]

giocatori_ddw = dcc.Dropdown(
    id="giocatori-column",
    options=[
        {"label": name, "value": code}
        for name, code in zip(
            sorted_players_list.GiocatoreName, sorted_players_list.index
        )
    ],
    value=sorted_players_list.index,
    multi=True,
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

partite_ddw = dcc.Dropdown(
    id="partite-column",
    options=[{"label": i, "value": i} for i in partite],
    value=partite,
    multi=True,
)

tipologia_ddw = dcc.Dropdown(
    id="tipologia-column",
    options=[{"label": i, "value": i} for i in tipo_attacchi],
    value=tipo_attacchi_filtrato,
    multi=True,
)

attack_after_ddw = dcc.Dropdown(
    id="attack-after-column",
    options=[
        {"label": fondamentale_testo, "value": fondamentale_codice}
        for fondamentale_testo, fondamentale_codice in zip(
            attack_after_testo, attack_after_codice
        )
    ],
    value=[fondamentale_codice for fondamentale_codice in attack_after_codice],
    multi=True,
)

set_by_ddw = dcc.Dropdown(
    id="set_by-column",
    options=[{"label": i, "value": i} for i in set_by],
    value=set_by[set_by.isin(setters_list)],
    multi=True,
)

rotazioni_ddw = dcc.Dropdown(
    id="rotazioni-column",
    options=[{"label": i, "value": i} for i in rotazioni],
    value=rotazioni,
    multi=True,
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

grouping_time_player_rbn = dcc.Dropdown(
    id="group-time-player",
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

ruoli_ddw = dcc.Dropdown(
    id="ruoli-column",
    options=[{"label": i, "value": i} for i in ruoli],
    value=ruoli,
    multi=True,
)

giocatore_singolo_ddw = dcc.Dropdown(
    id="giocatore-singolo-column",
    options=[
        {"label": name, "value": code}
        for name, code in zip(players_list.GiocatoreName, players_list.index)
    ],
    value=players_list.index[0],
    clearable=False,
)

giocatore_timeline_ddw = dcc.Dropdown(
    id="giocatore-timeline-column",
    options=[
        {"label": name, "value": code}
        for name, code in zip(players_list.GiocatoreName, players_list.index)
    ],
    value=players_list.index[0],
    clearable=False,
)


parametri_prestazione_ddw = dcc.Dropdown(
    id="prestazione-column",
    options=[{"label": i, "value": i} for i in traces],
    value="Kill",
    clearable=False,
)

# output = dcc.Graph(id='XvsY-Pass')

timeline_attack_plot = dcc.Graph(style={"height": "600px"}, id="timeline-attack-plot")
timeline_attack_plot_player = dcc.Graph(
    style={"height": "600px"}, id="timeline-attack-plot-player"
)

stats_lollipop_attack = dcc.Graph(id="stats-lollipop-attack")

stats_lollipop_attack_single_player = dcc.Graph(
    id="stats-lollipop-attack-single-player"
)

stats_winlose_attack = dcc.Graph(
    style={"height": "600px"}, id="stats-winlose-bar-attack"
)

pad = 4


# LAYOUT
def layout_attack():
    layout_attack = (
        html.Div(
            [
                dcc.Store(id="df-single-team-value"),
                dcc.Store(id="df-filtered-value"),
                dbc.Container(
                    [
                        dbc.Row(
                            [dbc.Col(html.H1("Attack performance"), className="mb-2")]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H6(
                                        children="Visualize attack stats from the South Korea National Team database"
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
                                            children="Team attack performance",
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
                        dbc.Row([dbc.Col([timeline_attack_plot], lg=12, md=14, xs=12)]),
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
                                        id="table-attack-team",
                                        columns=[
                                            dict(name="Team", id="SquadraName"),
                                            dict(name="Total", id="total"),
                                            dict(
                                                name="Kill (%)",
                                                id="kill",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Blocked (%)",
                                                id="blocked_perc",
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
                                            {"column_id": "total", "direction": "desc"}
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
                                        id="table-attack-team-container",
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
                                            children="Player attack performance",
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
                                    html.Label("Player:", className="float-right"),
                                    className="mt-4",
                                    lg=4,
                                    md=5,
                                    xs=12,
                                ),
                                dbc.Col(giocatore_timeline_ddw, className="mt-4"),
                            ],
                            className="row g-3 align-items-center",
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
                                dbc.Col(grouping_time_player_rbn, className="mt-4"),
                            ],
                            className="row g-3 align-items-center",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(timeline_attack_plot_player),
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
                                            children="Players comparison",
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
                                    html.Div(stats_lollipop_attack), lg=12, md=12, xs=12
                                ),
                            ],
                            className="mt-4",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dash_table.DataTable(
                                        id="table-attack",
                                        columns=[
                                            dict(name="Player", id="GiocatoreName"),
                                            dict(name="Total", id="total"),
                                            dict(
                                                name="Kill (%)",
                                                id="kill",
                                                type="numeric",
                                                format=Format(
                                                    precision=1, scheme=Scheme.fixed
                                                ),
                                            ),
                                            dict(
                                                name="Blocked (%)",
                                                id="blocked_perc",
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
                                            {"column_id": "total", "direction": "desc"}
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
                                        id="table-attack-container",
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
                                            children="Sets won/lost, by attack performance",
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
                                    html.Div(stats_winlose_attack),
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
                                            children="Individual player performance, based on position",
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
                        html.Div(
                            [
                                dbc.Col(
                                    html.Label("Player:", className="float-right"),
                                    className="mt-4",
                                    lg=4,
                                    md=5,
                                    xs=12,
                                ),
                                dbc.Col(giocatore_singolo_ddw, className="mt-4"),
                            ],
                            className="row g-3 align-items-center",
                        ),
                        dbc.Row(
                            [
                                # dbc.Col(html.Div(giocatore_singolo_ddw), lg=3, md=6, xs=12, style={'padding-top': 20}),
                                dbc.Col(
                                    html.Div(stats_lollipop_attack_single_player),
                                    lg=12,
                                    md=12,
                                    xs=12,
                                ),
                            ],
                            className="mt-4",
                        ),

                    ]
                ),
            ]
        ),
    )
    return layout_attack


# CALLBACK FUNCTIONS


@app.callback(Output("df-single-team-value", "data"), [Input("teams-column", "value")])
def select_team_data(selected_team):
    dff = df.loc[df["SquadraCodice"] == selected_team]
    players_df = players_list.loc[players_list.SquadraCodice == selected_team]

    sorted_players_list = get_sorted_players_list(players_df, dff)
    datasets = {
        "df_selected_team": dff.to_json(orient="split", date_format="iso"),
        "players_df": sorted_players_list.to_json(orient="split", date_format="iso"),
        # 'df_other_teams': df_other.to_json(orient='split', date_format='iso'),
    }

    return json.dumps(datasets)


@app.callback(
    [
        # Output('giocatori-column', 'options'),
        Output("giocatore-timeline-column", "options"),
        # Output('ruoli-column', 'options'),
        # Output('partite-column', 'options'),
        # Output('rotazioni-column', 'options'),
        # Output('tipologia-column', 'options'),
        # Output('set_by-column', 'options'),
        Output("giocatore-singolo-column", "options"),
        # Output('giocatori-column', 'value'),
        Output("giocatore-timeline-column", "value"),
        # Output('ruoli-column', 'value'),
        # Output('partite-column', 'value'),
        # Output('rotazioni-column', 'value'),
        # Output('tipologia-column', 'value'),
        # Output('set_by-column', 'value'),
        Output("giocatore-singolo-column", "value"),
    ],
    [Input("df-single-team-value", "data")],
)
def update_controls_options_and_values(df_single_team):
    datasets = json.loads(df_single_team)
    dff = pd.read_json(datasets["df_selected_team"], orient="split")
    players_df = pd.read_json(datasets["players_df"], orient="split")

    giocatori_opt = [
        {"label": name, "value": code}
        for name, code in zip(players_df.GiocatoreName, players_df.index)
    ]

    giocatori_val = players_df.index

    ruoli = dff["CurrentPlayerPosition"].unique()
    ruoli_opt = [{"label": i, "value": i} for i in ruoli]
    ruoli_val = ruoli

    partite = dff["Partita"].unique()
    partite_opt = [{"label": i, "value": i} for i in partite]
    partite_val = partite

    rotazioni = dff["iz"].unique()
    rotazioni_opt = [{"label": i, "value": i} for i in rotazioni]
    rotazioni_val = rotazioni

    tipo_attacchi = dff.type.unique()
    tipo_opt = [{"label": i, "value": i} for i in tipo_attacchi]
    tipo_val = tipo_attacchi_filtrato

    set_by = dff["SecondTouchGiocatoreName"].value_counts().reset_index()["index"]
    set_by_opt = [{"label": i, "value": i} for i in set_by]
    setters_code_list = dff["SetCode"].unique()
    setters_list = players_df["GiocatoreName"].loc[setters_code_list]
    set_by_val = set_by[set_by.isin(setters_list)]

    return giocatori_opt, giocatori_opt, giocatori_val[0], giocatori_val[0]
    # return giocatori_opt, giocatori_opt, ruoli_opt, partite_opt, rotazioni_opt, tipo_opt, set_by_opt, giocatori_opt, \
    #        giocatori_val, giocatori_val[0], ruoli_val, partite_val, rotazioni_val, tipo_val, set_by_val, giocatori_val[
    #            0]


@app.callback(
    Output("stats-lollipop-attack", "figure"), [Input("teams-column", "value")]
)
def update_lollipop_attack(team_code):
    fund = dvstats.Fundamental.attack
    gran = dvstats.Granularity.players
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
    [
        Output("table-attack-team", "data"),
        Output("table-attack-team", "style_data_conditional"),
    ],
    [
        Input("table-attack-team", "page_current"),
        Input("table-attack-team", "page_size"),
        Input("table-attack-team", "sort_by"),
        Input("table-attack-team", "filter_query"),
        Input("teams-column", "value"),
    ],
)
def update_table(page_current, page_size, sort_by, filter, code):
    dff = dvstats.get_stats_table(
        df, dvstats.Fundamental.attack, dvstats.Granularity.teams, teams_list
    )
    return do_table_filtering(
        dff, page_current, page_size, sort_by, filter
    ), colorize_attack_columns(dff)


@app.callback(
    [Output("table-attack", "data"), Output("table-attack", "style_data_conditional")],
    [
        Input("table-attack", "page_current"),
        Input("table-attack", "page_size"),
        Input("table-attack", "sort_by"),
        Input("table-attack", "filter_query"),
        Input("teams-column", "value"),
    ],
)
def update_table(page_current, page_size, sort_by, filter, code):
    dff = dvstats.get_stats_table(
        df[df.SquadraCodice == code],
        dvstats.Fundamental.attack,
        dvstats.Granularity.players,
        players_list,
    )
    return do_table_filtering(
        dff, page_current, page_size, sort_by, filter
    ), colorize_attack_columns(dff)


@app.callback(
    Output("timeline-attack-plot", "figure"),
    [
        Input("df-filtered-value", "data"),
        Input("group-time", "value"),
        Input("teams-column", "value"),
    ],
)
def update_timeline_graph_attack(df_filtered, radio_index, selected_team_code):
    all_teams = dvstats.get_teams_list(df)
    teams_df, selected_field_name, mode = dvstats.get_team_timeline_stats(
        radio_index, all_teams, df, dvstats.Fundamental.attack
    )
    return dvstats.get_timeline_plot_team(
        teams_df,
        selected_team_code,
        selected_field_name,
        mode,
        dvstats.Fundamental.attack,
        ["kill", "efficiency"],
    )


@app.callback(
    Output("timeline-attack-plot-player", "figure"),
    [
        Input("df-filtered-value", "data"),
        Input("group-time-player", "value"),
        Input("giocatore-timeline-column", "value"),
    ],
)
def update_timeline_graph_attack_player(df_filtered, radio_index, selected_player):
    all_players = players_list
    filtered_players_df, positions = dvstats.filter_relevant_positions_players_data(
        df, selected_player
    )
    players_df, selected_field_name, mode = dvstats.get_player_timeline_stats(
        radio_index, all_players, filtered_players_df, dvstats.Fundamental.attack
    )
    return dvstats.get_timeline_plot_player(
        players_df,
        selected_player,
        selected_field_name,
        mode,
        positions,
        dvstats.Fundamental.attack,
        ["kill", "efficiency"],
    )


@app.callback(
    Output("stats-lollipop-attack-single-player", "figure"),
    [
        Input("df-filtered-value", "data"),
        Input("giocatore-singolo-column", "value"),
    ],
)
def update_lollipop_attack_single_player(df_filtered, giocatore_code):
    fund = dvstats.Fundamental.attack
    all_players = players_list
    filtered_players_df, positions = dvstats.filter_relevant_positions_players_data(
        df, giocatore_code
    )
    stats = dvstats.get_stats_by_position(
        filtered_players_df, fund, dvstats.Granularity.players, all_players
    )[0]
    fig = dvstats.update_lollipop_single_player(
        stats, giocatore_code, dvstats.plots_viz_dict[fund]["performance"], fund
    )
    return fig


@app.callback(
    Output("stats-winlose-bar-attack", "figure"),
    [Input("df-filtered-value", "data"), Input("teams-column", "value")],
)
def update_bin_plot_attack(data, selected_team_code):
    fund = dvstats.Fundamental.attack
    return dvstats.get_binned_winning_stats_plot(
        df[df.SquadraCodice == selected_team_code], fund, "kill", 11
    )

