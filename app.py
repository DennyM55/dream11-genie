import streamlit as st
from groq import APITimeoutError, AuthenticationError, RateLimitError

from agent import pick_team, pick_team_demo, rate_limit_message
from data import SQUADS


def _pick_team_safely(match, mode):
    try:
        picker = pick_team_demo if mode == "Simple demo" else pick_team
        return picker(match, verbose=True)
    except AuthenticationError:
        st.error("The Groq API key is invalid or missing.")
    except RateLimitError as error:
        st.error(rate_limit_message(error))
    except APITimeoutError:
        st.error("Groq took too long to respond. No retry was attempted.")
    except ValueError:
        st.error("The generated team could not satisfy all Dream11 rules.")
    except Exception as error:
        print(f"[streamlit] team generation failed: {type(error).__name__}")
        st.error("Groq returned an invalid response. No automatic retry was attempted.")
    return None


st.set_page_config(page_title="Dream11 Genie", page_icon="🏏", layout="centered")

st.title("🏏 Dream11 Genie")
st.caption("An educational fantasy-XI demo using local sample cricket data.")

with st.form("team_generator"):
    match = st.selectbox("Pick a sample fixture", list(SQUADS.keys()))
    mode = st.segmented_control(
        "Mode",
        ["Simple demo", "Agent learning mode"],
        default="Simple demo",
        help=(
            "Simple demo uses one Groq request. Agent learning mode preserves the "
            "LangGraph tool-calling workflow and can use several requests."
        ),
    )
    submitted = st.form_submit_button("Generate my XI", type="primary")

st.info("All squads, form notes, pitch details, and player statistics shown here are sample data.")

if submitted:
    message = "Generating from local sample data with one Groq request..."
    if mode == "Agent learning mode":
        message = "The learning agent is calling sample-data tools and reasoning..."
    with st.spinner(message):
        team = _pick_team_safely(match, mode)

    if team is None:
        st.stop()

    st.subheader(f"Your sample-data XI — {team.match}")
    c1, c2 = st.columns(2)
    c1.metric("Captain (2x)", team.captain)
    c2.metric("Vice-captain (1.5x)", team.vice_captain)

    st.table(
        [
            {
                "Selection": "Captain" if p.player == team.captain else (
                    "Vice-captain" if p.player == team.vice_captain else ""
                ),
                "Player": p.player,
                "Role": p.role,
                "Team": p.team,
                "Why": p.reason,
            }
            for p in team.players
        ]
    )
    st.info(f"**Strategy:** {team.strategy}")
