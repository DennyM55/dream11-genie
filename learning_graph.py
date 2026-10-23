"""Minimal LangGraph workflow used by Agent learning mode."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent import pick_team_demo
from schema import Team
from validate import repair_team, validate_team


class TeamState(TypedDict, total=False):
    match: str
    verbose: bool
    team: Team


def generate_team(state: TeamState) -> TeamState:
    """AI node: reuse the working single-request generation path."""
    return {
        "team": pick_team_demo(state["match"], verbose=state.get("verbose", False))
    }


def validate_and_repair(state: TeamState) -> TeamState:
    """Python node: enforce all Dream11 rules without another model call."""
    team = state["team"]
    issues, _ = validate_team(team)
    if issues:
        team = repair_team(team)

    remaining, _ = validate_team(team)
    if remaining:
        raise ValueError("Could not produce a valid XI:\n" + "\n".join(remaining))
    return {"team": team}


def _build_graph():
    builder = StateGraph(TeamState)
    builder.add_node("generate_team", generate_team)
    builder.add_node("validate_and_repair", validate_and_repair)
    builder.add_edge(START, "generate_team")
    builder.add_edge("generate_team", "validate_and_repair")
    builder.add_edge("validate_and_repair", END)
    return builder.compile()


TEAM_GRAPH = _build_graph()


def pick_team(match: str, *, verbose: bool = False) -> Team:
    """Run Agent learning mode and return its validated Team."""
    result = TEAM_GRAPH.invoke({"match": match, "verbose": verbose})
    return result["team"]
