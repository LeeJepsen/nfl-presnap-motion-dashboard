# using streamlit for NFL PreSnap Dashboard
# Big Data Bowl 2025 data weeks 1-9
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NFL Pre-snap Motion Dashboard",
    layout="wide"
)

st.title("NFL Pre-snap Motion Dashboard")
st.markdown(
    """
    #### League Overview — Weeks 1–9
    Interactive exploration of league-wide pre-snap motion tendencies and defensive adjustment
    behavior.
    """
)

# Paths 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Loading in team data
import base64
import requests

@st.cache_data
def load_team_branding():
    url = "https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/teams_colors_logos.csv"
    teams = pd.read_csv(url)

    keep = []
    for col in ["team_abbr", "team_color", "team_color2", "team_logo_espn", "team_logo_wikipedia", "team_wordmark"]:
        if col in teams.columns:
            keep.append(col)
    teams = teams[keep].copy()

    # Pick a logo column that exists
    logo_col = None
    for candidate in ["team_logo_espn", "team_logo_wikipedia", "team_wordmark"]:
        if candidate in teams.columns:
            logo_col = candidate
            break

    teams = teams.rename(columns={"team_abbr": "team", logo_col: "logo_url"})
    return teams, logo_col

teams_branding, _logo_col_used = load_team_branding()


@st.cache_data
def fetch_image_as_base64(url: str) -> str | None:
    """Download an image and return as base64 data URI for Plotly."""
    if not isinstance(url, str) or not url.startswith("http"):
        return None
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        b64 = base64.b64encode(r.content).decode("utf-8")
        # Most are png; if not, Plotly usually still renders it, but png is safest.
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def add_logos_to_bar(fig, df, y_col="MotionRate"):
    """
    Places a small logo near the top of each bar.
    Works best when x is categorical (team abbreviations).
    """
    # Size tuning (paper units)
    sizex = 0.20  # width in "category space" - small
    sizey = 0.20  # height in y-axis units? no, because yref='paper'/'y' combo affects it.
    # We'll use yref='y' and sizey in y units; so scale it based on your y range.
    # We'll compute a logo height as ~3% of y-range.
    y_max = float(df[y_col].max()) if len(df) else 1.0
    logo_height = max(0.01, 0.03 * y_max)  # 3% of max

    for _, row in df.iterrows():
        team = row["offenseTeam"]
        y_val = float(row[y_col])

        img_uri = fetch_image_as_base64(row.get("logo_url"))
        if img_uri is None:
            continue

        # Place logo slightly below the bar top (inside)
        y_logo = max(0, y_val - (logo_height * 1.2))

        fig.add_layout_image(
            dict(
                source=img_uri,
                x=team,
                y=y_logo,
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="bottom",
                sizex=0.6,          # categorical sizing is quirky; 0.5–0.8 tends to work
                sizey=logo_height,
                sizing="contain",
                opacity=0.95,
                layer="above",
            )
        )

    return fig


# Load data

@st.cache_data
def load_data():
    """
    Loads the 3 small files Tab 1 needs.
    These should be in: data/processed/
      - MotionSummaryWeeks_1_9.csv  (computed in notebook)
      - games.csv
      - plays.csv
    """
    ms = pd.read_csv(DATA_PROCESSED / "MotionSummaryWeeks_1_9.csv")
    games = pd.read_csv(DATA_PROCESSED / "games.csv")
    plays = pd.read_csv(DATA_PROCESSED / "plays.csv")
    return ms, games, plays

MotionSummaryAll, games, plays = load_data()


# Static defenders threshold + figure (weeks 1–9 overall)

@st.cache_data
def compute_defender_curve_and_cross(ms_all: pd.DataFrame, bin_width: float = 1.0, min_n: int = 25):
    """
    Returns (x, y, cross_x) where:
      x = mean motion distance per bin
      y = P(success) per bin, success := num_defenders_moved_gt >= 2
      cross_x = interpolated x where y crosses 0.5 (None if not found)
    """
    df = ms_all.dropna(subset=["motion_distance", "num_defenders_moved_gt"]).copy()
    if df.empty:
        return None, None, None

    df["Success"] = (df["num_defenders_moved_gt"] >= 2).astype(int)

    # Build bins from 0 to 13, pulled from data exploration
    max_x = 13
    bins = np.arange(0, max_x + bin_width, bin_width)
    df["bin"] = pd.cut(df["motion_distance"], bins=bins, include_lowest=True, right=False)

    g = df.groupby("bin", observed=True).agg(
        x=("motion_distance", "mean"),
        y=("Success", "mean"),
        n=("Success", "size")
    ).reset_index().sort_values("x")

    g = g[g["n"] >= min_n].copy()
    if g.empty:
        return None, None, None

    x = g["x"].to_numpy()
    y = g["y"].to_numpy()

    # Find crossing where y crosses 0.5
    threshold_prob = 0.5
    cross_x = None
    for i in range(1, len(x)):
        if (y[i-1] < threshold_prob and y[i] >= threshold_prob):
            # linear interpolation between (x[i-1], y[i-1]) and (x[i], y[i])
            x0, y0 = x[i-1], y[i-1]
            x1, y1 = x[i], y[i]
            if y1 != y0:
                cross_x = x0 + (threshold_prob - y0) * (x1 - x0) / (y1 - y0)
            else:
                cross_x = x1
            break

    return x, y, cross_x


def build_defenders_shift_matplotlib(x, y, cross_x, threshold_prob: float = 0.5):
    if x is None or y is None:
        return None

    fig = plt.figure(figsize=(10, 5))

    # main curve
    plt.plot(x, y, linewidth=2.5)

    # 50% reference line
    plt.axhline(threshold_prob, linestyle="--", linewidth=1)

    # label the 50% line directly
    plt.text(
        0.2, threshold_prob + 0.02,
        "50% defensive adjustment likelihood",
        va="bottom"
    )

    # shaded region where adjustment is likely (>= 50%)
    above = y >= threshold_prob
    plt.fill_between(
        x, y, threshold_prob,
        where=above,
        interpolate=True,
        alpha=0.11
    )

    # annotate the crossing point
    if cross_x is not None and np.isfinite(cross_x):
        plt.scatter([cross_x], [threshold_prob], s=40)
        plt.text(
            cross_x,
            threshold_prob - 0.07,
            f"{cross_x:.2f} yards",
            ha="center",
            va="top",
            fontweight="bold"
        )

    plt.xlim(0, 13)
    plt.ylim(0, 1.0)

    plt.xlabel("Motion distance (yards)")
    plt.ylabel("Probability(≥2 defenders moved > 1.5y)")
    plt.title("Motion Distance and Defensive Adjustment Likelihood (Weeks 1–9)")

    # styling: spines + grid + ticks
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.3
    )
    ax.set_axisbelow(True)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis="x", which="minor", length=3)

    plt.tight_layout()
    return fig

# Compute static defender response curve (once)
X_DEF, Y_DEF, CROSS_X = compute_defender_curve_and_cross(MotionSummaryAll)

# Build matplotlib figure
DEF_SHIFT_FIG = build_defenders_shift_matplotlib(X_DEF, Y_DEF, CROSS_X)


# Tabs

tab1, = st.tabs(["League Overview"])

with tab1:
    # Controls row 
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        start_wk, end_wk = st.slider(
            "Week Range",
            min_value=1,
            max_value=9,
            value=(1, 9),
            step=1
        )

    with c2:
        sort_mode = st.selectbox(
            "Sort teams by",
            ["Motion Rate", "Motion Plays", "Total Plays", "Team (A→Z)"],
            index=0
        )

    with c3:
        top_n = st.number_input(
            "Top N teams",
            min_value=8,
            max_value=32,
            value=32,
            step=1
        )

    st.divider()

    # Build plays2 (plays + week) 
    plays_with_week = plays.merge(games[["gameId", "week"]], on="gameId", how="left")
    plays2 = plays_with_week[["gameId", "playId", "possessionTeam", "defensiveTeam", "week"]].copy()

    # Filter by week slider 
    PlaysRange = plays2[plays2["week"].between(start_wk, end_wk)].copy()
    MSFilt = MotionSummaryAll[MotionSummaryAll["week"].between(start_wk, end_wk)].copy()

    # Handle teams that have had a bye
    GamesRange = games[games["week"].between(start_wk, end_wk)].copy()

    # count games played by each team
    gp_home = GamesRange.groupby("homeTeamAbbr").size().reset_index(name="GamesPlayed")
    gp_away = GamesRange.groupby("visitorTeamAbbr").size().reset_index(name="GamesPlayed")

    games_played = pd.concat([
        gp_home.rename(columns={"homeTeamAbbr": "team"}),
        gp_away.rename(columns={"visitorTeamAbbr": "team"})
    ], ignore_index=True)

    games_played = games_played.groupby("team")["GamesPlayed"].sum().reset_index()

    # Team motion rate table 
    total_plays = (
        PlaysRange.groupby("possessionTeam")
        .size()
        .reset_index(name="TotalOffPlays")
        .rename(columns={"possessionTeam": "offenseTeam"})
    )

    motion_plays = (
        MSFilt.groupby("offenseTeam")
        .size()
        .reset_index(name="MotionPlays")
    )

    # calc motion rate also league average 
    MotionRate = total_plays.merge(motion_plays, on="offenseTeam", how="left")
    MotionRate = MotionRate.merge(games_played, left_on = "offenseTeam", right_on = "team",
                                  how = "left").drop(columns=["team"])
    MotionRate["GamesPlayed"] = MotionRate["GamesPlayed"].fillna(0).astype(int)
    max_gp = MotionRate["GamesPlayed"].max()
    MotionRate["MotionPlays"] = MotionRate["MotionPlays"].fillna(0).astype(int)
    MotionRate["MotionRate"] = MotionRate["MotionPlays"] / MotionRate["TotalOffPlays"]
    league_avg = MotionRate["MotionRate"].mean()
    MotionRate["AvgGroup"] = np.where(
    MotionRate["MotionRate"] >= league_avg,
    "Above Avg",
    "Below Avg"
)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)

    league_motion_rate = len(MSFilt) / len(PlaysRange)

    avg_motion_dist = MSFilt["motion_distance"].mean()

    def_adj_rate = (MSFilt["num_defenders_moved_gt"] >= 2).mean()

    avg_motion_plays = MSFilt.groupby("offenseTeam").size().mean()

    k1.metric("League Motion Rate", f"{league_motion_rate:.1%}")
    k2.metric("Avg Motion Distance", f"{avg_motion_dist:.2f} yd")
    k3.metric("Defensive Adjustment Rate", f"{def_adj_rate:.1%}")
    k4.metric("Avg Motion Plays / Team", f"{avg_motion_plays:.1f}")

    

    with st.expander("About these metrics"):
        st.markdown("""
    **League Motion Rate**  
    Percentage of offensive plays that included pre-snap motion within the selected week range.

    **Avg Motion Distance (yards)**  
    Average distance traveled by the motion player prior to the snap.  
    Distances are calculated from player tracking coordinates during identified motion windows.

    **Defensive Adjustment Rate**  
    Percentage of motion plays where **at least two defenders moved ≥ 1.5 yards** during the motion window.  
    This serves as a proxy for meaningful defensive response.

    **Avg Motion Plays per Team**  
    Average number of motion plays executed per offense within the selected week range.  
    This helps contextualize differences caused by bye weeks or uneven game counts.

    **Data Source**  
    NFL Big Data Bowl 2025 tracking data (Weeks 1–9). Motion events and defender movement were derived from player tracking frames prior to ball snap.
    """)

    st.divider()
    
    # Sorting for bar chart
    df_bar = MotionRate.copy()

    if sort_mode == "Motion Rate":
        df_bar = df_bar.sort_values("MotionRate", ascending=False)
    elif sort_mode == "Motion Plays":
        df_bar = df_bar.sort_values("MotionPlays", ascending=False)
    elif sort_mode == "Total Plays":
        df_bar = df_bar.sort_values("TotalOffPlays", ascending=False)
    else:
        df_bar = df_bar.sort_values("offenseTeam", ascending=True)

    df_bar = df_bar.head(int(top_n)).copy()

st.subheader("Motion Rate by Offense")

# adding in so the teams color will show from df
df_bar = df_bar.merge(
    teams_branding[["team", "team_color", "logo_url"]],
    left_on="offenseTeam",
    right_on="team",
    how="left"
)
# Fallback if a team color is missing
df_bar["team_color"] = df_bar["team_color"].fillna("#1f77b4")


# Drop down menu changes axis
if sort_mode == "Motion Rate":
    y_col = "MotionRate"
    y_label = "Motion Rate (%)"

elif sort_mode == "Motion Plays":
    y_col = "MotionPlays"
    y_label = "Motion Plays"

elif sort_mode == "Total Plays":
    y_col = "TotalOffPlays"
    y_label = "Total Offensive Plays"

else:  # Team (A→Z)
    y_col = "MotionRate"
    y_label = "Motion Rate"

with st.container(border=True):

    if MotionRate.empty or pd.isna(league_avg):
        st.info("Not enough data in this week range to compute the league average line.")
    else:
        # Adding team color and logos
        color_map = dict(zip(df_bar["offenseTeam"], df_bar["team_color"]))
        fig_bar = px.bar(
            df_bar,
            x="offenseTeam",
            y=y_col,
            color="offenseTeam",
            color_discrete_map=color_map,
            hover_data={
                "GamesPlayed": True,
                "MotionRate": ":.1%",
                "MotionPlays": True,
                "TotalOffPlays": True,
                "offenseTeam": False
            },
            labels={"offenseTeam": "Team", y_col: y_label}
        )
        fig_bar.update_yaxes(title_text=y_label)
        fig_bar = add_logos_to_bar(fig_bar, df_bar, y_col=y_col)

        # Add avg line only for motion rate
        if y_col == "MotionRate" and not pd.isna(league_avg):
            fig_bar.add_hline(y=league_avg, line_dash="dash", line_width=1, line_color="gray")

        # Dim only when plotting Motion Rate
        if y_col == "MotionRate":

            for trace in fig_bar.data:
                team = trace.name

                group = df_bar.loc[
                    df_bar["offenseTeam"] == team, "AvgGroup"
                ].iloc[0]

                if group == "Below Avg":
                    trace.opacity = 0.45   # dimmed
                else:
                    trace.opacity = 1.0

        # Editing layout
        fig_bar.update_layout(
            xaxis_title_font=dict(color="black"),
            yaxis_title_font=dict(color="black"),
            xaxis_tickfont=dict(color="black"),
            yaxis_tickfont=dict(color="black")
        )
        fig_bar.update_xaxes(
            showline=True,
            linecolor="black",
            ticks="outside",
            tickcolor="black"
        )
        fig_bar.update_yaxes(
            showline=True,
            linecolor="black",
            ticks="outside",
            tickcolor="black"
        )
        # Centered annotation and Add average reference line
        if y_col == "MotionRate" and not pd.isna(league_avg):    
            fig_bar.add_hline(
                y=league_avg,
                line_dash="dash",
                line_width=1,
                line_color="black"
            )

            fig_bar.add_annotation(
                x=0.8,                     
                y=league_avg,
                xref="paper",              
                yref="y",
                text=f"League Avg: {league_avg:.1%}",
                showarrow=False,
                font=dict(size=11, color="black"),
                yshift=10
            )

        fig_bar.update_layout(
            xaxis_tickangle=-45,
            showlegend=False
        )
    
        for trace in fig_bar.data:
            team = trace.name
            gp = df_bar.loc[df_bar["offenseTeam"] == team, "GamesPlayed"].iloc[0]
            if gp < max_gp:
                trace.opacity = 0.65  # slightly dim teams with fewer games

        if y_col == "MotionRate":
            fig_bar.update_yaxes(tickformat=".0%")
        else:
            fig_bar.update_yaxes(tickformat=",")

        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Bottom row: histogram + static insight
    left, right = st.columns([2, 1.35])

    with left:
        with st.container(border=True):
            st.subheader("Motion Distance Distribution")
            if MSFilt.empty:
                st.info("No motion plays in this week range.")
            else:
                fig_hist = px.histogram(
                    MSFilt,
                    x="motion_distance",
                    nbins=30,
                    labels={"motion_distance": "Motion Distance (yards)"}
                )
                fig_hist.update_xaxes(
                    dtick=5,         
                    ticks="outside", 
                    ticklen=6,        
                    tickwidth=1,
                    tickcolor="black"
                )
                fig_hist.update_traces(
                    marker_line_width=0.6,
                    marker_line_color="rgba(0,0,0,0.6)"
                )
                fig_hist.update_layout(
                    xaxis_title_font=dict(color="black"),
                    yaxis_title_font=dict(color="black"),
                    xaxis_tickfont=dict(color="black"),
                    yaxis_tickfont=dict(color="black")
                )
                fig_hist.update_yaxes(title_text="Number of Pre-Snap Motion Plays")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                

        with right:
            with st.container(border=True):
                st.subheader(
                        "League-wide Defender Response (Static)",
                        help="""
                        Shows how the probability of a meaningful defensive adjustment
                        (≥2 defenders moving ≥1.5 yards) increases as pre-snap motion
                        distance increases across the league (Weeks 1–9).

                        The marked point indicates the motion distance where a defensive
                        adjustment becomes ~50% likely.
                        """
                        )

                if CROSS_X is None:
                    st.write("Not enough data to estimate the 50% threshold reliably.")
                else:
                    st.metric("50% threshold distance", f"{CROSS_X:.2f} yards")

                if DEF_SHIFT_FIG is None:
                    st.info("Could not build defender-shift plot (check data/columns).")
                else:
                    st.pyplot(DEF_SHIFT_FIG, clear_figure=True)

                st.caption("Cumulative Measure (weeks 1-9) This panel intentionally does NOT change when the week slider changes.")

    # show the underlying table 
    with st.expander("Show team table"):
        st.dataframe(MotionRate.sort_values("MotionRate", ascending=False), use_container_width=True)