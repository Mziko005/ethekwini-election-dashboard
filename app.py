import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="eThekwini Election Analytics",
    page_icon="📊",
    layout="wide"
)

DATA = Path(__file__).parent / "data"

@st.cache_data
def load_data(filename):
    path = DATA / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

st.title("eThekwini Election Analytics")
st.caption(
    "Historical IEC Election Evidence | "
    "2026 Conditional Model Scenarios"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Historical Elections",
        "2026 Party Estimates",
        "Three Ward Predictions",
        "Voter Turnout",
        "Model Evaluation",
        "Limitations"
    ]
)

historical = load_data("historical_pr.csv")
metro = load_data("conditional_metro_2026.csv")
wards = load_data("conditional_ward_leaders_2026.csv")
turnout = load_data("historical_turnout.csv")
turnout_scenarios = load_data("turnout_scenarios_2026.csv")
regression = load_data("regression_comparison.csv")
classification = load_data("classification_comparison.csv")

if page == "Overview":

    st.header("Municipal Election Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Municipality", "eThekwini")
    c2.metric("Historical Elections", "3")
    c3.metric("Prediction Year", "2026")

    st.info(
        "Historical results cover 2011, 2016 and 2021. "
        "The 2026 results are conditional model scenarios, "
        "not official election outcomes."
    )

    if turnout is not None:
        fig = px.line(
            turnout,
            x="Year",
            y="TurnoutPercentage",
            markers=True,
            title="Historical IEC Voter Turnout"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Historical Elections":

    st.header("Historical PR Election Results")

    if historical is None:
        st.warning("Historical PR data are missing.")
    else:
        election_years = sorted(
            historical["Year"].dropna().unique().tolist()
        )

        year = st.selectbox(
            "Select election year",
            election_years
        )

        df = historical[
            historical["Year"] == year
        ].copy()

        party_votes = (
            df.groupby(
                "Party_Standardised",
                as_index=False
            )["TotalValidVotes"].sum()
        )

        total = party_votes["TotalValidVotes"].sum()

        if total > 0:
            party_votes["VoteShare"] = (
                party_votes["TotalValidVotes"] / total * 100
            )

            party_votes = party_votes.sort_values(
                "VoteShare", ascending=False
            )

            fig = px.bar(
                party_votes.head(15),
                x="Party_Standardised",
                y="VoteShare",
                title=f"Party Support — {year} PR Ballot",
                labels={
                    "Party_Standardised": "Political Party",
                    "VoteShare": "Vote Share (%)"
                }
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(party_votes, use_container_width=True)

elif page == "2026 Party Estimates":

    st.header("2026 Conditional Metro Scenario")

    st.warning(
        "Preliminary model scenario using historical "
        "2021 parties and district vote weights. "
        "Not a complete validated 2026 election forecast."
    )

    if metro is None:
        st.warning("Metro prediction file is missing.")
    else:
        fig = px.bar(
            metro.head(15),
            x="Party_Standardised",
            y="EstimatedShare2026",
            title="Conditional Estimated Party Shares",
            labels={
                "Party_Standardised": "Party",
                "EstimatedShare2026": "Estimated Share (%)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metro, use_container_width=True)

elif page == "Three Ward Predictions":

    st.header("Three Selected Wards")

    st.warning(
        "Ward predictions are exploratory scenarios. "
        "Historical geographic alignment remains unverified."
    )

    if wards is None:
        st.warning("Ward prediction file is missing.")
    else:
        st.dataframe(wards, use_container_width=True)

        fig = px.bar(
            wards,
            x="Ward",
            y="EstimatedShare2026",
            color="ProjectedLeader",
            title="Estimated Leading Party Share by Ward"
        )

        st.plotly_chart(fig, use_container_width=True)

elif page == "Voter Turnout":

    st.header("Voter Turnout Analysis")

    if turnout is not None:
        st.subheader("Historical Turnout")

        fig = px.bar(
            turnout,
            x="Year",
            y="TurnoutPercentage",
            title="Historical IEC Turnout Rates",
            labels={
                "TurnoutPercentage": "Turnout (%)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(turnout, use_container_width=True)

    if turnout_scenarios is not None:
        st.subheader("2026 Conditional Scenarios")

        st.info(
            "Scenario counts assume the 2021 effective "
            "registration denominator remains unchanged. "
            "They are not calibrated prediction intervals."
        )

        st.dataframe(
            turnout_scenarios,
            use_container_width=True
        )

elif page == "Model Evaluation":

    st.header("Machine Learning Model Comparison")

    if regression is not None:
        st.subheader("Regression Models")
        st.dataframe(regression, use_container_width=True)

        fig = px.bar(
            regression,
            x="Model",
            y="MAE",
            title="Regression Model MAE"
        )

        st.plotly_chart(fig, use_container_width=True)

    if classification is not None:
        st.subheader("Classification Models")
        st.dataframe(
            classification,
            use_container_width=True
        )

        fig = px.bar(
            classification,
            x="Model",
            y="F1",
            title="Classification Model F1-score"
        )

        st.plotly_chart(fig, use_container_width=True)

elif page == "Limitations":

    st.header("Methodology and Limitations")

    st.markdown("""
    ### Historical Data
    The study uses IEC municipal election records from
    2011, 2016 and 2021.

    ### Model Limitations
    - Only three municipal election cycles are available.
    - Voting-district identifiers may not represent
      unchanged geographic boundaries.
    - The historical matched-data approach excludes
      some party and district combinations.
    - New 2026 political parties are not fully represented.
    - Ward model transfer has not been independently
      validated.
    - Conditional turnout scenarios use historical
      registration assumptions.

    ### Interpretation
    All 2026 outputs are preliminary model scenarios.

    **Source:** Electoral Commission of South Africa.
    https://results.elections.org.za/home/Downloads/ME-Results
    """)
