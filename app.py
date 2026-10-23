import streamlit as st
from groq import APITimeoutError, AuthenticationError, RateLimitError

from agent import pick_team
from data import SQUADS


def _pick_team_safely(match):
    try:
        return pick_team(match, verbose=True)
    except AuthenticationError:
        st.error("The Groq API key is invalid or missing.")
    except RateLimitError:
        st.error("The Groq token limit has been reached. Please try again later.")
    except APITimeoutError:
        st.error("Groq took too long to respond. Please try again.")
    except ValueError:
        st.error("The generated team could not satisfy all Dream11 rules.")
    except Exception as error:
        print(f"[streamlit] team generation failed: {type(error).__name__}")
        st.error("Groq returned an invalid response. Please try again.")
    return None


st.set_page_config(page_title="Dream11 Genie", page_icon="🏏", layout="centered")

st.title("🏏 Dream11 Genie")
st.caption("An AI agent that reasons over live-style data to pick your fantasy XI.")

match = st.selectbox("Pick a fixture", list(SQUADS.keys()))

if st.button("Generate my XI", type="primary"):
    with st.spinner("The agent is calling tools and reasoning…"):
        team = _pick_team_safely(match)

    if team is None:
        st.stop()

    st.subheader(f"Your XI — {team.match}")
    c1, c2 = st.columns(2)
    c1.metric("Captain (2x)", team.captain)
    c2.metric("Vice-captain (1.5x)", team.vice_captain)

    st.table(
        [
            {
                "": "⭐" if p.player == team.captain else ("🔸" if p.player == team.vice_captain else ""),
                "Player": p.player,
                "Role": p.role,
                "Team": p.team,
                "Why": p.reason,
            }
            for p in team.players
        ]
    )
    st.info(f"**Strategy:** {team.strategy}")
