
"""Dream11 Genie — the agent itself.

A ReAct agent powered by Groq (free tier), built with LangChain's create_agent
(which runs on LangGraph underneath). It reasons about a fixture,
calls the tools to gather live-style data, then returns a structured XI with a
captain, vice-captain and a reason for every pick.

Needs a free Groq API key in the environment as GROQ_API_KEY (see .env.example).
"""
from __future__ import annotations

import os
import json
from pydantic import ValidationError
from dotenv import load_dotenv, find_dotenv
from groq import BadRequestError, GroqError
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
import truststore

from data import PLAYERS, SQUADS
from schema import Team
from tools import TOOLS, get_pitch_report, search_form
from validate import validate_team, player_pool_text, repair_credits, repair_team

# Find a .env by walking up from the current directory (so it works whether the
# key sits in this project folder or a parent course folder).
load_dotenv(find_dotenv(usecwd=True))

# Use the operating system's trusted certificate store. This keeps TLS
# verification enabled while supporting local Windows networks whose trusted
# root certificate is not included in Python's bundled CA file.
truststore.inject_into_ssl()

# Groq (free tier) with LangGraph. openai/gpt-oss-120b is Groq's flagship
# tool-use model. Override via GROQ_MODEL (e.g. "openai/gpt-oss-20b" for speed).
# Needs GROQ_API_KEY in the environment — get a free key at https://console.groq.com/keys
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_PARSE_ATTEMPTS = 1

DEMO_SYSTEM = """You select a fantasy cricket XI from supplied SAMPLE data.
Return ONLY one JSON object, without markdown or commentary, with keys match, captain, vice_captain, players, strategy.
players must be an array of objects with player, team, role, reason string keys.
Use only listed players and copy the supplied team/role values exactly.
Obey: 11 unique players; WK 1-4, BAT 3-6, AR 1-4, BOWL 3-6;
at most 100 credits; at most 7 players per real team; captain and vice-captain
must be different members of the XI. Keep each reason to one short sentence."""

SYSTEM = """You are Dream11 Genie, an expert fantasy cricket analyst.

Your job: given a fixture, pick the BEST possible fantasy XI using the data
provided by your tools. Do not invent players or statistics.

WORKFLOW — follow these steps IN ORDER:
1. Call get_match_squads to learn the available players and venue.
2. Call get_pitch_report for that venue.
3. Call search_form for BOTH teams.
4. Call get_player_stats for 2-3 key captaincy candidates.
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


def _make_llm() -> ChatGroq:
    """Build a Groq model with settings supported by GPT-OSS tool use."""
    return ChatGroq(
        model=MODEL,
        temperature=0,
        reasoning_format="hidden",
        max_tokens=4096,
        timeout=60,
        max_retries=0,
        model_kwargs={"parallel_tool_calls": False},
    )


def _make_demo_llm() -> ChatGroq:
    """Build the no-retry model used for the single-request demo path."""
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        reasoning_format="hidden",
        max_tokens=2000,
        timeout=60,
        max_retries=0,
    )


def rate_limit_message(error: BaseException) -> str:
    """Turn Groq's actual 429 details into a useful, non-retrying UI message."""
    body = getattr(error, "body", None)
    details = body.get("error", {}) if isinstance(body, dict) else {}
    raw = " ".join(
        str(value)
        for value in (details.get("code"), details.get("type"), details.get("message"), error)
        if value
    ).lower()

    if "tokens per day" in raw or "tpd" in raw or "daily token" in raw:
        return "Groq's daily token quota is exhausted. It will reset later; no retry was attempted."
    if "tokens per minute" in raw or "tpm" in raw:
        return "Groq's per-minute token limit was reached. Wait for the limit window to reset; no retry was attempted."
    if "requests per day" in raw or "rpd" in raw or "daily request" in raw:
        return "Groq's daily request quota is exhausted. It will reset later; no retry was attempted."
    if "requests per minute" in raw or "rpm" in raw or "request limit" in raw:
        return "Groq's request-rate limit was reached. Wait for the limit window to reset; no retry was attempted."
    return "Groq returned a rate-limit error. Check the provider limit details; no retry was attempted."


def _groq_error_detail(error: BaseException) -> str:
    """Return safe provider error metadata without request headers or secrets."""
    body = getattr(error, "body", None)
    details = body.get("error", {}) if isinstance(body, dict) else {}
    code = details.get("code") or "unknown"
    error_type = details.get("type") or type(error).__name__
    message = details.get("message") or str(error)
    return f"code={code}, type={error_type}, message={message}"


def _is_parse_failure(error: BaseException) -> bool:
    if not isinstance(error, BadRequestError):
        return False
    body = getattr(error, "body", None)
    details = body.get("error", {}) if isinstance(body, dict) else {}
    return details.get("code") in {
        "output_parse_failed",
        "json_validate_failed",
        "tool_use_failed",
    }


class _SafeTraceHandler(BaseCallbackHandler):
    """Print stages and tool names only; never prompts, results, or headers."""

    def __init__(self, stage: str):
        self.stage = stage
        self.model_turn = 0
        self.tool_names = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.model_turn += 1
        print(f"   [{self.stage}] model turn {self.model_turn} started")

    def on_llm_end(self, response, *, run_id, **kwargs):
        message = response.generations[0][0].message
        tool_names = [call.get("name", "unknown") for call in message.tool_calls]
        finish = message.response_metadata.get("finish_reason", "unknown")
        print(f"   [{self.stage}] model turn completed: finish={finish}, tools={tool_names}")

    def on_llm_error(self, error, *, run_id, **kwargs):
        print(f"   [{self.stage}] model turn failed: {_groq_error_detail(error)}")

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "unknown")
        self.tool_names[run_id] = name
        print(f"   [{self.stage}] tool started: {name}")

    def on_tool_end(self, output, *, run_id, **kwargs):
        print(f"   [{self.stage}] tool completed: {self.tool_names.pop(run_id, 'unknown')}")

    def on_tool_error(self, error, *, run_id, **kwargs):
        name = self.tool_names.pop(run_id, "unknown")
        print(f"   [{self.stage}] tool failed: {name} ({type(error).__name__})")


def _make_agent():
    """Build a fresh agent instance (cheap — no network call until invoke)."""
    return create_agent(
        model=_make_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM,
    )


def _invoke_agent(agent, messages, stage: str, verbose: bool):
    """Invoke a tool agent, retrying only Groq output parsing failures."""
    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        if verbose:
            print(f"   [{stage}] attempt {attempt}/{MAX_PARSE_ATTEMPTS}")
        config = {"callbacks": [_SafeTraceHandler(stage)]} if verbose else None
        try:
            return agent.invoke({"messages": messages}, config=config)
        except BadRequestError as error:
            if not _is_parse_failure(error) or attempt == MAX_PARSE_ATTEMPTS:
                raise RuntimeError(
                    f"Groq failed during {stage} after {attempt} attempt(s): "
                    f"{_groq_error_detail(error)}"
                ) from error
            if verbose:
                print(f"   [{stage}] recoverable parse failure; retrying")
    raise RuntimeError(f"Groq failed during {stage}")


def _structured_team(messages, instruction: str, stage: str, verbose: bool) -> Team:
    """Convert a completed tool conversation to a Pydantic Team."""
    structured_llm = _make_llm().with_structured_output(
        Team,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        if verbose:
            print(f"   [{stage}] attempt {attempt}/{MAX_PARSE_ATTEMPTS}")
        try:
            response = structured_llm.invoke(messages + [("user", instruction)])
        except BadRequestError as error:
            if not _is_parse_failure(error) or attempt == MAX_PARSE_ATTEMPTS:
                raise RuntimeError(
                    f"Groq failed during {stage} after {attempt} attempt(s): "
                    f"{_groq_error_detail(error)}"
                ) from error
            if verbose:
                print(f"   [{stage}] recoverable parse failure; retrying")
            continue

        team = response.get("parsed")
        if team is not None:
            if verbose:
                print(f"   [{stage}] structured Team completed")
            return team

        raw = response.get("raw")
        finish_reason = getattr(raw, "response_metadata", {}).get("finish_reason")
        parsing_error = response.get("parsing_error")
        if attempt == MAX_PARSE_ATTEMPTS:
            detail = f"; parsing error: {parsing_error}" if parsing_error else ""
            raise RuntimeError(
                f"Groq did not return a structured Team during {stage} after "
                f"{attempt} attempt(s) (finish reason: {finish_reason or 'unknown'})"
                f"{detail}"
            )
        if verbose:
            print(f"   [{stage}] missing structured Team; retrying")

    raise RuntimeError(f"Groq failed during {stage}")


def _format_issues(issues: list[str]) -> str:
    return "\n".join(f"- {issue}" for issue in issues)


def _demo_prompt(match: str) -> str:
    """Build compact local evidence for the one-request demo mode."""
    fixture = SQUADS.get(match)
    if fixture is None:
        raise ValueError(f"No sample data is available for {match}")

    pitch = get_pitch_report(fixture["venue"])
    teams = list(fixture["teams"])
    player_lines = []
    for team_code in teams:
        for name in fixture["teams"][team_code]:
            player = PLAYERS[name]
            player_lines.append(
                f"{name}|{team_code}|{player['role']}|{player['credits']}cr|"
                f"avg fantasy {player['avg_fantasy_points']}|{player['note']}"
            )

    return (
        f"Fixture: {match}\nVenue: {fixture['venue']}\n"
        f"Pitch: {pitch['surface']}; {pitch['pace_vs_spin']}\n"
        f"Sample form: {' | '.join(f'{code}: {search_form(code)}' for code in teams)}\n"
        "Sample player data (name|team|role|credits|summary):\n"
        + "\n".join(player_lines)
    )


def _parse_demo_response(response) -> Team:
    """Parse plain JSON locally; never retry or replace unusable AI output."""
    if response.response_metadata.get("finish_reason") != "stop":
        raise ValueError("Groq returned an incomplete response. No retry was attempted.")
    content = response.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned an empty response. No retry was attempted.")
    content = content.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()
    try:
        team = Team.model_validate(json.loads(content))
    except (ValueError, ValidationError) as error:
        raise ValueError("Groq returned malformed JSON or missing team fields. No retry was attempted.") from error
    if not team.players or not any(p.player in PLAYERS for p in team.players):
        raise ValueError("Groq returned no usable sample players. No retry was attempted.")
    return team


def pick_team_demo(match: str, *, verbose: bool = False) -> Team:
    """Generate and locally validate a Team with exactly one Groq request.

    Local sample tools are called directly before the request. Any rule-invalid
    Team is repaired deterministically; provider and parsing failures are not
    retried.
    """
    prompt = _demo_prompt(match)  # Reject unknown fixtures before any API request.
    response = _make_demo_llm().invoke(
        [("system", DEMO_SYSTEM), ("user", prompt)]
    )
    team = _parse_demo_response(response)
    team = team.model_copy(update={"match": match})
    issues, _ = validate_team(team)
    if issues:
        if verbose:
            print(f"[demo] repairing {len(issues)} validation issue(s) locally")
        team = repair_team(team)
    final_issues, _ = validate_team(team)
    if final_issues:
        raise ValueError("Could not produce a valid XI:\n" + _format_issues(final_issues))
    return team


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
    result = _invoke_agent(
        agent,
        [("user", f"Pick the best Dream11 fantasy XI for {match}. "
          "Use all your tools, follow the hard rules, and explain every pick.")],
        "tool gathering",
        verbose,
    )

    team = _structured_team(
        result["messages"],
        "Return your final XI now as a Team matching the required schema. "
        "Use only the cricket evidence already gathered above.",
        "structured team generation",
        verbose,
    )
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
    try:
        corrected = _invoke_agent(
            agent,
            result["messages"] + [("user", correction)],
            "correction tool gathering",
            verbose,
        )

        team = _structured_team(
            corrected["messages"],
            "Return the corrected XI now as a Team matching the required schema. "
            "Fix every validation violation listed above.",
            "corrected structured team generation",
            verbose,
        )
        issues, stats = validate_team(team)

        if verbose:
            print(f"\n📋 Corrected draft: {len(team.players)} players | "
                  f"{stats['credits']} credits | issues={len(issues)}")
            for issue in issues:
                print(f"   ⚠ {issue}")
    except (GroqError, RuntimeError) as error:
        if verbose:
            print(
                "   [correction] provider correction unavailable; "
                f"using deterministic repair ({type(error).__name__})"
            )

    if not issues:
        return team

    # --- Pass 3: deterministic rule repair ---------------------------------
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
        if verbose:
            print("\nRunning deterministic full-team repair...")
        try:
            team = repair_team(team)
        except ValueError as error:
            raise ValueError(
                f"Could not produce a valid XI for {match}: {error}"
            ) from error
        issues, stats = validate_team(team)

        if verbose:
            print(f"   After full repair: {stats['credits']} credits | "
                  f"issues={len(issues)}")

    if issues:
        raise ValueError(
            f"Could not produce a valid XI for {match}:\n"
            + _format_issues(issues)
        )

    return team
