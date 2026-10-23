
"""Dream11 Genie — the agent itself.

A ReAct agent powered by Groq (free tier), built with LangChain's create_agent
(which runs on LangGraph underneath). It reasons about a fixture,
calls the tools to gather live-style data, then returns a structured XI with a
captain, vice-captain and a reason for every pick.

Needs a free Groq API key in the environment as GROQ_API_KEY (see .env.example).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from schema import Team
from tools import TOOLS
from validate import validate_team, player_pool_text, repair_credits

# Find a .env by walking up from the current directory (so it works whether the
# key sits in this project folder or a parent course folder).
load_dotenv(find_dotenv(usecwd=True))

# Groq (free tier) with LangGraph. openai/gpt-oss-120b is Groq's flagship
# tool-use model. Override via GROQ_MODEL (e.g. "openai/gpt-oss-20b" for speed).
# Needs GROQ_API_KEY in the environment — get a free key at https://console.groq.com/keys
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM = """You are Dream11 Genie, an expert fantasy cricket analyst.

Your job: given a fixture, pick the BEST possible fantasy XI using the data
provided by your tools. Do not invent players or statistics.

WORKFLOW — follow these steps IN ORDER:
1. Call get_match_squads to learn the available players and venue.
2. Call get_pitch_report for that venue.
3. Call search_form for BOTH teams.
4. Call get_player_stats for at least 8-10 key players (preferably all 11+).
5. Reason about form, pitch fit, role balance and fantasy points.
6. Output your final XI.

HARD DREAM11 RULES — you MUST satisfy ALL:
- Exactly 11 unique players, all from the two squads.
- Roles: WK 1-4, BAT 3-6, AR 1-4, BOWL 3-6.
- Total credits <= 100.0.
- Max 7 players from any one team.
- Captain and vice-captain must be DIFFERENT and BOTH in the XI.
- Captain scores 2x, vice-captain 1.5x — favour consistent high scorers.

SELECTION PHILOSOPHY:
- Recent form > reputation. Don't pick a star just for the name.
- Pitch suitability matters: on a turner, favour spinners and spin-hitters.
- Prefer all-rounders who contribute in both innings.
- For captain/VC, favour players with a high, consistent points floor.
- Every pick needs a one-line reason grounded in the tool data.
- Strategy should explain the overall logic in 2-3 sentences.

Be decisive. Use the tools, reason, then commit to a team."""


def _make_agent():
    """Build a fresh agent instance (cheap — no network call until invoke)."""
    llm = ChatGroq(model=MODEL, temperature=0)
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM,
        response_format=Team,
    )


def _format_issues(issues: list[str]) -> str:
    return "\n".join(f"- {issue}" for issue in issues)


def pick_team(match: str, *, verbose: bool = False) -> Team:
    """Pick a validated fantasy XI for a fixture.

    The agent gathers data via tools, produces a structured Team, and we run
    deterministic checks. If invalid, we ask for ONE correction pass.
    Credit overruns are repaired in Python as a last resort.

    Raises ValueError if a valid XI cannot be produced after correction.
    """
    agent = _make_agent()

    if verbose:
        print(f"\n🏏 Dream11 Genie | {match} | model={MODEL}")
        print("=" * 55)

    # --- Pass 1: agent picks a team ----------------------------------------
    result = agent.invoke({
        "messages": [("user", f"Pick the best Dream11 fantasy XI for {match}. "
                      "Use all your tools, follow the hard rules, and explain every pick.")]
    })

    team: Team = result["structured_response"]
    issues, stats = validate_team(team)

    if verbose:
        print(f"\n📋 First draft: {len(team.players)} players | "
              f"{stats['credits']} credits | issues={len(issues)}")
        for issue in issues:
            print(f"   ⚠ {issue}")

    if not issues:
        return team

    # --- Pass 2: one correction attempt -----------------------------------
    # Give the model its mistakes AND the exact player pool so it can fix them.
    correction = (
        f"Your proposed XI for {match} FAILED validation.\n\n"
        f"VIOLATIONS:\n{_format_issues(issues)}\n\n"
        f"VALID PLAYER POOL (name — team, role, credits):\n"
        f"{player_pool_text(match)}\n\n"
        "Fix ALL violations. Return a corrected XI with exactly 11 players. "
        "Keep your original strategic intent where possible, but obey every "
        "hard rule. Captain and vice-captain must be in the XI."
    )

    if verbose:
        print("\n🔄 Asking agent to correct the XI...")

    # Continue the SAME conversation so the agent retains its tool observations.
    # We append the correction to the existing messages, not start from scratch.
    corrected = agent.invoke({
        "messages": result["messages"] + [("user", correction)]
    })

    team = corrected["structured_response"]
    issues, stats = validate_team(team)

    if verbose:
        print(f"\n📋 Corrected draft: {len(team.players)} players | "
              f"{stats['credits']} credits | issues={len(issues)}")
        for issue in issues:
            print(f"   ⚠ {issue}")

    if not issues:
        return team

    # --- Pass 3: deterministic credit repair -------------------------------
    # Numeric constraints are better enforced in code than left to the LLM.
    if any("credits" in issue for issue in issues):
        if verbose:
            print("\n🔧 Running deterministic credit repair...")
        team = repair_credits(team)
        issues, stats = validate_team(team)

        if verbose:
            print(f"   After repair: {stats['credits']} credits | "
                  f"issues={len(issues)}")
            for issue in issues:
                print(f"   ⚠ {issue}")

    if issues:
        raise ValueError(
            f"Could not produce a valid XI for {match}:\n"
            + _format_issues(issues)
        )

    return team
