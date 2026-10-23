
"""Offline tests for Dream11 Genie.

Run with: python test_offline.py
No Groq key or network access required.
"""
from __future__ import annotations

import ssl
import os
from unittest.mock import Mock
from langchain_core.messages import AIMessage
from unittest.mock import patch

import agent
from agent import _make_llm, pick_team_demo, rate_limit_message
from data import PLAYERS, SQUADS
from schema import Pick, Team
from tools import get_match_squads, get_player_stats, search_form, get_pitch_report
from validate import validate_team, repair_credits, repair_team


def _make_team(names, captain=None, vice_captain=None):
    """Build a Team from known player names for testing."""
    picks = [
        Pick(
            player=n,
            team=PLAYERS[n]["team"],
            role=PLAYERS[n]["role"],
            reason="Test selection",
        )
        for n in names
    ]
    return Team(
        match="CSK vs MI",
        captain=captain or names[0],
        vice_captain=vice_captain or names[1],
        players=picks,
        strategy="Offline test team",
    )


def test_tools():
    squads = get_match_squads("CSK vs MI")
    assert "teams" in squads
    assert len(squads["teams"]["CSK"]) > 0
    assert len(squads["teams"]["MI"]) > 0

    assert get_player_stats("Ruturaj Gaikwad")["role"] == "BAT"
    assert get_player_stats("Gaikwad")["team"] == "CSK"
    assert "error" in get_player_stats("Unknown Player")

    assert isinstance(search_form("CSK"), str)
    assert "surface" in get_pitch_report("Chepauk")


def test_groq_client_safety_settings():
    with patch.dict(os.environ, {"GROQ_API_KEY": "offline-test-only"}):
        llm = _make_llm()
        demo = agent._make_demo_llm()
    assert demo.max_retries == 0
    assert demo.max_tokens == 2000
    assert "response_format" not in demo.model_kwargs
    assert "tools" not in demo.model_kwargs
    assert ssl.SSLContext.__module__ == "truststore._api"
    assert llm.request_timeout == 60
    assert llm.max_retries == 0
    assert llm.model_kwargs["parallel_tool_calls"] is False


def test_demo_uses_one_request_and_repairs_locally():
    names = [
        "MS Dhoni", "Ruturaj Gaikwad", "Devon Conway", "Ajinkya Rahane",
        "Shivam Dube", "Ravindra Jadeja", "Deepak Chahar",
        "Matheesha Pathirana", "Maheesh Theekshana", "Tushar Deshpande",
        "Jasprit Bumrah",
    ]
    invalid_team = _make_team(names)
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=invalid_team.model_dump_json(), response_metadata={"finish_reason": "stop"})
    with patch("agent._make_demo_llm", return_value=llm) as make_llm:
        repaired = pick_team_demo("CSK vs MI")
    make_llm.assert_called_once_with()
    llm.invoke.assert_called_once()
    issues, meta = validate_team(repaired)
    assert not issues
    assert meta["teams"] == {"CSK": 7, "MI": 4}


def test_rate_limit_messages_use_provider_error_details():
    class ProviderError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.body = {"error": {"code": "rate_limit_exceeded", "message": message}}

    assert "per-minute token" in rate_limit_message(ProviderError("tokens per minute (TPM)"))
    assert "daily token" in rate_limit_message(ProviderError("tokens per day (TPD)"))
    assert "request-rate" in rate_limit_message(ProviderError("requests per minute (RPM)"))
    assert "daily request" in rate_limit_message(ProviderError("requests per day (RPD)"))


def test_valid_team():
    names = [
        "MS Dhoni",
        "Ajinkya Rahane",
        "Devon Conway",
        "Suryakumar Yadav",
        "Ravindra Jadeja",
        "Shivam Dube",
        "Hardik Pandya",
        "Jasprit Bumrah",
        "Matheesha Pathirana",
        "Maheesh Theekshana",
        "Piyush Chawla",
    ]
    team = _make_team(names)
    issues, meta = validate_team(team)
    assert not issues, f"Expected valid XI, got {issues}"
    assert meta["credits"] <= 100


def test_duplicate_player():
    names = [
        "MS Dhoni", "MS Dhoni",
        "Ruturaj Gaikwad", "Devon Conway", "Suryakumar Yadav",
        "Ravindra Jadeja", "Shivam Dube", "Hardik Pandya",
        "Jasprit Bumrah", "Matheesha Pathirana", "Piyush Chawla",
    ]
    team = _make_team(names)
    issues, _ = validate_team(team)
    assert any("duplicate" in issue for issue in issues)


def test_over_budget():
    names = [
        "Ruturaj Gaikwad",
        "MS Dhoni",
        "Devon Conway",
        "Suryakumar Yadav",
        "Ravindra Jadeja",
        "Shivam Dube",
        "Hardik Pandya",
        "Jasprit Bumrah",
        "Matheesha Pathirana",
        "Maheesh Theekshana",
        "Rohit Sharma",
    ]
    team = _make_team(names, captain="Ruturaj Gaikwad", vice_captain="Suryakumar Yadav")
    issues, _ = validate_team(team)
    assert any("credits" in issue for issue in issues)

    repaired = repair_credits(team)
    new_issues, meta = validate_team(repaired)
    assert meta["credits"] <= 100
    assert not new_issues, f"Repair failed: {new_issues}"


def test_wrong_roles():
    names = [
        "MS Dhoni", "Ishan Kishan",
        "Ruturaj Gaikwad", "Devon Conway", "Suryakumar Yadav",
        "Ravindra Jadeja", "Shivam Dube", "Hardik Pandya",
        "Moeen Ali", "Tilak Varma", "Tim David",
    ]
    team = _make_team(names)
    issues, _ = validate_team(team)
    assert any("BOWL" in issue for issue in issues)


def test_team_cap():
    names = [
        "MS Dhoni", "Ruturaj Gaikwad", "Devon Conway",
        "Shivam Dube", "Ravindra Jadeja", "Moeen Ali",
        "Deepak Chahar", "Matheesha Pathirana",
        "Suryakumar Yadav", "Jasprit Bumrah", "Hardik Pandya",
    ]
    team = _make_team(names)
    issues, _ = validate_team(team)
    assert any("max 7" in issue for issue in issues)


def test_repair_ten_players_from_one_team():
    names = [
        "MS Dhoni", "Ruturaj Gaikwad", "Devon Conway", "Ajinkya Rahane",
        "Shivam Dube", "Ravindra Jadeja", "Deepak Chahar",
        "Matheesha Pathirana", "Maheesh Theekshana", "Tushar Deshpande",
        "Jasprit Bumrah",
    ]
    team = _make_team(names)
    original_roles = validate_team(team)[1]["roles"]
    issues, meta = validate_team(team)
    assert meta["teams"]["CSK"] == 10
    assert any("max 7" in issue for issue in issues)

    repaired = repair_team(team)
    new_issues, new_meta = validate_team(repaired)
    assert not new_issues, f"Team-balance repair failed: {new_issues}"
    assert new_meta["teams"] == {"CSK": 7, "MI": 4}
    assert new_meta["roles"] == original_roles
    assert new_meta["credits"] <= 100
    assert len({pick.player for pick in repaired.players}) == 11
    assert repaired.captain in {pick.player for pick in repaired.players}
    assert repaired.vice_captain in {pick.player for pick in repaired.players}
    assert repaired.captain != repaired.vice_captain


def test_correction_provider_failure_uses_deterministic_repair():
    names = [
        "MS Dhoni", "Ruturaj Gaikwad", "Devon Conway", "Ajinkya Rahane",
        "Shivam Dube", "Ravindra Jadeja", "Deepak Chahar",
        "Matheesha Pathirana", "Maheesh Theekshana", "Tushar Deshpande",
        "Jasprit Bumrah",
    ]
    invalid_team = _make_team(names)

    with (
        patch("agent._make_agent", return_value=object()),
        patch(
            "agent._invoke_agent",
            side_effect=[{"messages": []}, RuntimeError("provider unavailable")],
        ),
        patch("agent._structured_team", return_value=invalid_team),
    ):
        repaired = agent.pick_team("CSK vs MI")

    issues, meta = validate_team(repaired)
    assert not issues
    assert meta["teams"] == {"CSK": 7, "MI": 4}


def test_captain_not_in_xi():
    names = [
        "MS Dhoni", "Ruturaj Gaikwad", "Devon Conway",
        "Suryakumar Yadav", "Ravindra Jadeja", "Shivam Dube",
        "Hardik Pandya", "Jasprit Bumrah", "Matheesha Pathirana",
        "Maheesh Theekshana", "Piyush Chawla",
    ]
    team = _make_team(names, captain="Rohit Sharma")
    issues, _ = validate_team(team)
    assert any("captain is not" in issue for issue in issues)


def _valid_sample():
    return _make_team([
        "MS Dhoni", "Ajinkya Rahane", "Devon Conway", "Suryakumar Yadav",
        "Ravindra Jadeja", "Shivam Dube", "Hardik Pandya", "Jasprit Bumrah",
        "Matheesha Pathirana", "Maheesh Theekshana", "Piyush Chawla",
    ])


def test_demo_json_parsing():
    team = _valid_sample()
    for content in (team.model_dump_json(), "```json\n" + team.model_dump_json() + "\n```"):
        parsed = agent._parse_demo_response(AIMessage(content=content, response_metadata={"finish_reason": "stop"}))
        assert parsed == team
    for content, finish in [("{", "stop"), ("{}", "stop"), ("[]", "stop"), ("", "stop"), (team.model_dump_json(), "length")]:
        try:
            agent._parse_demo_response(AIMessage(content=content, response_metadata={"finish_reason": finish}))
        except ValueError:
            pass
        else:
            raise AssertionError("Unusable output accepted")
    team.players = []
    try:
        agent._parse_demo_response(AIMessage(content=team.model_dump_json(), response_metadata={"finish_reason": "stop"}))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty XI fabricated into a team")


def test_demo_repairs_metadata_and_captaincy():
    team = _valid_sample()
    team.players[0].role = "BAT"
    team.players[0].team = "MI"
    team.vice_captain = team.captain
    assert validate_team(team)[0]
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=team.model_dump_json(), response_metadata={"finish_reason": "stop"})
    with patch("agent._make_demo_llm", return_value=llm):
        repaired = pick_team_demo("CSK vs MI")
    assert not validate_team(repaired)[0]
    assert len(repaired.players) == 11
    assert repaired.players[0].role == PLAYERS[repaired.players[0].player]["role"]
    llm.invoke.assert_called_once()


def test_demo_provider_error_never_retries():
    import httpx
    from groq import RateLimitError
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    error = RateLimitError("daily tokens", response=response, body={"error": {"message": "tokens per day (TPD)"}})
    llm = Mock()
    llm.invoke.side_effect = error
    with patch("agent._make_demo_llm", return_value=llm):
        try:
            pick_team_demo("CSK vs MI")
        except RateLimitError as caught:
            assert caught is error
        else:
            raise AssertionError("Provider error hidden")
    llm.invoke.assert_called_once()
    assert "daily token" in rate_limit_message(error)


def test_streamlit_success_and_safe_errors():
    import httpx
    from groq import AuthenticationError, RateLimitError, APITimeoutError
    from streamlit.testing.v1 import AppTest
    team = _valid_sample()
    with patch.dict(os.environ, {"GROQ_API_KEY": "offline-test-only"}):
        app = AppTest.from_file("app.py").run()
        assert not app.exception
        assert app.segmented_control[0].value == "Simple demo"
        with patch("agent.pick_team_demo", return_value=team) as picker:
            app.button[0].click().run()
        picker.assert_called_once()
        assert not app.exception and not app.error
        assert len(app.table[0].value) == 11
        assert "Credits" in app.table[0].value.columns
        assert len(app.metric) == 3
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        errors = [
            AuthenticationError("sensitive", response=httpx.Response(401, request=request), body={}),
            RateLimitError("sensitive", response=httpx.Response(429, request=request), body={"error": {"message": "tokens per day"}}),
            APITimeoutError(request=request),
            ValueError("sensitive malformed JSON"),
        ]
        for error in errors:
            with patch("agent.pick_team_demo", side_effect=error) as picker:
                app.button[0].click().run()
            picker.assert_called_once()
            assert not app.exception and app.error
            assert "sensitive" not in app.error[0].value
            assert not app.table


if __name__ == "__main__":
    tests = [
        test_demo_json_parsing,
        test_demo_repairs_metadata_and_captaincy,
        test_demo_provider_error_never_retries,
        test_streamlit_success_and_safe_errors,
        test_tools,
        test_groq_client_safety_settings,
        test_demo_uses_one_request_and_repairs_locally,
        test_rate_limit_messages_use_provider_error_details,
        test_valid_team,
        test_duplicate_player,
        test_over_budget,
        test_wrong_roles,
        test_team_cap,
        test_repair_ten_players_from_one_team,
        test_correction_provider_failure_uses_deterministic_repair,
        test_captain_not_in_xi,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} offline tests passed.")
