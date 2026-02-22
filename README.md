# NFL Pre-Snap Motion Dashboard

Interactive analytics dashboard built using **NFL Big Data Bowl 2025 tracking data** to explore how offensive pre-snap motion influences defensive behavior across the league.

This project combines player tracking data engineering, feature construction, and interactive visualization to translate raw tracking data into interpretable insights.

---

## Project Motivation

Player tracking is the next step in sports, and being able to interpret the data from a sport science perspective (defining player load) using gps systems has been the recent standard in athlete monitoring. Now technology has gotten to a point where we can utilize this for on field context. So this is my entry into working with NFL player tracking data.

Going with the NFL Big Data Bowl prompt of Pre-snap motion, which has become a defining feature of modern NFL offenses, and its league-wide impact on pre-snap reads, and defensive behavior.

This dashboard was developed to investigate:

- How frequently teams use pre-snap motion
- How aggressive motion is across the league
- When motion meaningfully forces defensive adjustment
- How teams compare relative to league tendencies

The objective is not only statistical analysis, but creating an **interactive decision-support tool** grounded in football context.

---

## Current Release — v1.0 (League Overview)

Version 1.0 introduces the **League Overview dashboard**, providing an interactive summary of pre-snap motion behavior across Weeks 1–9 of the NFL season.

### Features

- Interactive week-range filtering
- League motion KPI framework
- Team comparison visualization with dynamic metric selection
- League average benchmarking
- Context visual encoding
- Bye week aware comparisons
- Defensive adjustment probability modeling
- Static analytical insight panel derived from league-wide data

---

## Key Metrics

| Metric                        | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| **League Motion Rate**        | Percentage of offensive plays using pre-snap motion                |
| **Avg Motion Distance**       | Average distance traveled by the motion player (yards)             |
| **Defensive Adjustment Rate** | Percentage of motion plays producing meaningful defensive reaction |
| **Avg Motion Plays per Team** | Average motion volume per offense within selected weeks            |

These metrics summarize league tendencies (weeks 1-9 in NFL's 2022 nfl season)

---

## Defensive Adjustment Definition

A defensive adjustment is defined as **at least two defenders moving ≥ 1.5 yards during the motion window**.

This definition was not arbitrarily selected. It was derived through exploratory analysis performed in the project’s notebook workflow:

1. Defensive player displacement distributions were examined across all motion plays.
2. Small movements (< ~1 yard) were frequently observed even without meaningful reaction, representing normal pre-snap alignment behavior.
3. Movement thresholds were evaluated to separate routine positioning from coordinated defensive response.
4. Requiring movement from **multiple defenders** reduced noise from isolated player adjustments.
5. A 1.5-yard displacement threshold consistently identified moments where defensive structure visibly changed.

Together, these criteria operationalize a football concept — _defensive reaction_ — into a measurable tracking-data feature.

---

## Analytical Insight

League-wide aggregation shows that the probability of forcing defensive adjustment increases with motion distance, with a transition point occurring near the distance where defensive response becomes 50/50 (~50% probability).

This relationship forms the basis of the static insight visualization included in the dashboard.

---

## Data Pipeline

NFL Tracking Data → Motion Detection → Feature Engineering → Aggregation → Interactive Dashboard

### Processing Steps

- Identified motion windows using tracking event markers
- Calculated motion player displacement and duration
- Measured defender movement relative to motion
- Engineered play-level defensive response features
- Aggregated league summaries for dashboard consumption
- Exported processed datasets for reproducible visualization

All feature engineering and exploratory analysis were conducted in the project notebooks prior to dashboard development.

---

## Dashboard Preview

📄 **[View Full Dashboard Preview (PDF)](images/dashboard_snapshot.pdf)**

### Dashboard Functionality

The League Overview dashboard is designed as an interactive exploration tool rather than a static visualization. Key functionality includes:

#### Interactive Controls

- Adjustable **week range slider** allowing analysis across any subset of Weeks 1–9
- Dynamic filtering updates all KPIs and visualizations simultaneously
- Team comparison sorting options for multiple performance perspectives

#### League Motion KPIs

- League Motion Rate
- Average Motion Distance (yards)
- Defensive Adjustment Rate
- Average Motion Plays per Team (bye-week aware)

#### Team Comparison Visualization

- Dynamic bar chart comparing offensive motion tendencies
- Metric switching between:
  - Motion Rate
  - Motion Plays
  - Total Offensive Plays
- League-average benchmark indicator
- Context-aware dimming for below-average teams
- Team color branding and logo integration

#### Defensive Adjustment Analysis

- League-wide probability curve relating motion distance to defensive response
- Highlighted transition threshold where defensive adjustment becomes likely
- Static reference visualization derived from full-season aggregation

#### Data Context Handling

- Bye weeks accounted for through games-played normalization
- Metrics dynamically recomputed based on selected week range

#### User Experience Design

- Consistent axis scaling and labeling
- Interactive hover tooltips for detailed inspection
- Modular layout designed for future dashboard expansion

---

## Running the Application Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
