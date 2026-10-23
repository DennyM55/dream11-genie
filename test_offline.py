
"""Offline tests for Dream11 Genie.

Run with: python test_offline.py
No Groq key or network access required.
"""
from __future__ import annotations

from data import PLAYERS, SQUADS
from schema import Pick, Team
from tools import get_match_squads, get_player_stats, search_form, get_pitch_report
from validate import validate_team, repair_credits


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


def test_valid_team():
    names = [
        "MS Dhoni",
        "Ruturaj Gaikwad",
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


if __name__ == "__main__":
    tests = [
        test_tools,
        test_valid_team,
        test_duplicate_player,
        test_over_budget,
        test_wrong_roles,
        test_team_cap,
        test_captain_not_in_xi,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} offline tests passed.")
