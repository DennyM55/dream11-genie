import os
import streamlit as st
from groq import APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

from agent import pick_team, pick_team_demo, rate_limit_message
from data import PLAYERS, SQUADS
from validate import validate_team


def _pick_team_safely(match, mode):
    if not os.getenv("GROQ_API_KEY"):
        st.error("The Groq API key is missing. Configure GROQ_API_KEY in deployment secrets.")
        return None
    try:
        picker = pick_team_demo if mode == "Simple demo" else pick_team
        team = picker(match, verbose=False)
        if validate_team(team)[0]:
            raise ValueError("Invalid XI")
        return team
    except AuthenticationError as error:
        print(f"[streamlit] category=authentication status={getattr(error, 'status_code', 'unknown')}")
        st.error("The Groq API key is invalid or missing.")
    except RateLimitError as error:
        print(f"[streamlit] category=rate_limit status={getattr(error, 'status_code', 'unknown')}")
        st.error(rate_limit_message(error))
    except APITimeoutError as error:
        print(f"[streamlit] category=timeout type={type(error).__name__}")
        st.error("Groq took too long to respond. No retry was attempted.")
    except APIConnectionError as error:
        print(f"[streamlit] category=connection type={type(error).__name__}")
        st.error("Could not connect to Groq. No retry was attempted.")
    except BadRequestError as error:
        body = getattr(error, "body", None)
        details = body.get("error", {}) if isinstance(body, dict) else {}
        print(
            "[streamlit] category=http_400 status=400 "
            f"provider_code={details.get('code', 'unknown')} provider_type={details.get('type', 'unknown')}"
        )
        st.error("Groq rejected the generation request (HTTP 400). No retry was attempted.")
    except ValueError as error:
        print(f"[streamlit] category=invalid_output detail={error}")
        if "truncated" in str(error).lower() or "finish_reason=length" in str(error):
            st.error("Groq's response was truncated before the XI JSON completed. No retry was attempted.")
        else:
            st.error("Groq returned malformed, incomplete, or invalid team data. No retry was attempted.")
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

st.session_state.setdefault("generated_team", None)

if submitted:
    message = "Generating from local sample data with one Groq request..."
    if mode == "Agent learning mode":
        message = "The learning agent is calling sample-data tools and reasoning..."
    with st.spinner(message):
        team = _pick_team_safely(match, mode)
    st.session_state.generated_team = team

team = st.session_state.generated_team
if team is not None:
    st.subheader(f"Your sample-data XI — {team.match}")
    c1, c2 = st.columns(2)
    c1.metric("Captain (2x)", team.captain)
    c2.metric("Vice-captain (1.5x)", team.vice_captain)

    _, stats = validate_team(team)
    st.metric("Total credits (maximum 100)", stats["credits"])
    st.caption("Roles: " + " · ".join(f"{role}: {count}" for role, count in stats["roles"].items()))
    st.table(
        [
            {
                "Selection": "Captain" if p.player == team.captain else (
                    "Vice-captain" if p.player == team.vice_captain else ""
                ),
                "Player": p.player,
                "Role": p.role,
                "Team": p.team,
                "Credits": PLAYERS[p.player]["credits"],
                "Why": p.reason,
            }
            for p in team.players
        ]
    )
    st.info(f"**Strategy:** {team.strategy}")
