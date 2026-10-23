
"""Deterministic Dream11 rule validator.

LLMs reason well but slip on hard numeric/count constraints (credits, per-team
caps). We never trust the model to police those — we check them in code, and
feed any violations back for a single correction pass (see agent.pick_team).
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, product

from data import PLAYERS, SQUADS
from schema import Pick, Team

RULES = {
    "size": 11,
    "composition": {"WK": (1, 4), "BAT": (3, 6), "AR": (1, 4), "BOWL": (3, 6)},
    "max_credits": 100.0,
    "max_per_team": 7,
}


def validate_team(team) -> tuple[list[str], dict]:
    """Return (list of rule violations, summary stats). Empty list == valid."""
    issues: list[str] = []
    names = [p.player for p in team.players]
    known = [n for n in names if n in PLAYERS]

    unknown = [n for n in names if n not in PLAYERS]
    if unknown:
        issues.append(f"players not in the squad data: {unknown}")
    if len(names) != RULES["size"]:
        issues.append(f"must be {RULES['size']} players, got {len(names)}")
    if len(set(names)) != len(names):
        issues.append("duplicate players in the XI")

    fixture = SQUADS.get(team.match)
    if fixture is None:
        issues.append("unknown sample fixture")
    else:
        pool = {name for squad in fixture["teams"].values() for name in squad}
        if any(name not in pool for name in names):
            issues.append("players outside the selected fixture")
    for pick in team.players:
        if pick.player in PLAYERS:
            data = PLAYERS[pick.player]
            if pick.role != data["role"] or pick.team != data["team"]:
                issues.append(f"incorrect role or team for {pick.player}")

    roles = Counter(PLAYERS[n]["role"] for n in known)
    for role, (lo, hi) in RULES["composition"].items():
        c = roles.get(role, 0)
        if not (lo <= c <= hi):
            issues.append(f"{role} count is {c}, must be {lo}-{hi}")

    credits = round(sum(PLAYERS[n]["credits"] for n in known), 1)
    if credits > RULES["max_credits"]:
        issues.append(f"total credits {credits} exceed the {RULES['max_credits']} cap")

    teams = Counter(PLAYERS[n]["team"] for n in known)
    for t, c in teams.items():
        if c > RULES["max_per_team"]:
            issues.append(f"{c} players from {t} — max {RULES['max_per_team']} from one team")

    if team.captain not in names:
        issues.append("captain is not in the XI")
    if team.vice_captain not in names:
        issues.append("vice-captain is not in the XI")
    if team.captain == team.vice_captain:
        issues.append("captain and vice-captain are the same player")

    return issues, {"credits": credits, "roles": dict(roles), "teams": dict(teams)}


def repair_credits(team: Team) -> Team:
    """Deterministically bring an over-budget XI under the credit cap.

    Greedily swaps the priciest NON-captain players for the cheapest unused
    same-role player, never breaking the per-team cap. This is the 'hard numeric
    constraint belongs in code, not the LLM' pattern — guaranteed correct.
    Composition and per-team validity are preserved (same-role swaps only).
    """
    names = [p.player for p in team.players]
    reasons = {p.player: p.reason for p in team.players}
    original = {p.player: p for p in team.players}  # fall back to these for names not in squad data
    protected = {team.captain, team.vice_captain}

    def total(ns):
        return round(sum(PLAYERS[n]["credits"] for n in ns if n in PLAYERS), 1)

    def team_counts(ns):
        return Counter(PLAYERS[n]["team"] for n in ns if n in PLAYERS)

    guard = 0
    while total(names) > RULES["max_credits"] and guard < 30:
        guard += 1
        best = None  # (key, index, replacement_name) — key sorts by least quality lost
        for i, n in enumerate(names):
            if n in protected or n not in PLAYERS:
                continue
            role, cost = PLAYERS[n]["role"], PLAYERS[n]["credits"]
            for cand, cp in PLAYERS.items():
                if cand in names or cp["role"] != role:
                    continue
                trial = names.copy(); trial[i] = cand
                if any(v > RULES["max_per_team"] for v in team_counts(trial).values()):
                    continue
                saving = cost - cp["credits"]
                if saving <= 0:
                    continue
                # prefer the swap that loses the FEWEST fantasy points; break ties
                # toward the bigger credit saving (fewer total swaps needed).
                pts_loss = PLAYERS[n]["avg_fantasy_points"] - cp["avg_fantasy_points"]
                key = (pts_loss, -saving)
                if best is None or key < best[0]:
                    best = (key, i, cand)
        if not best:
            break
        _, i, cand = best
        old = names[i]
        names[i] = cand
        reasons[cand] = f"Budget swap in for {old} — {PLAYERS[cand]['note']}"

    # Rebuild from squad data for known players; for any name the model invented
    # that isn't in the squad data, keep its original Pick (repair never swaps
    # those, and looking them up in PLAYERS would KeyError). validate_team still
    # flags them afterward — we just don't crash on the way out.
    picks = []
    for n in names:
        if n in PLAYERS:
            picks.append(Pick(player=n, team=PLAYERS[n]["team"], role=PLAYERS[n]["role"],
                              reason=reasons.get(n, PLAYERS[n]["note"])))
        else:
            picks.append(original[n])
    return Team(match=team.match, captain=team.captain, vice_captain=team.vice_captain,
                players=picks, strategy=team.strategy)


def repair_team(team: Team) -> Team:
    """Rebuild an invalid XI from known squad players under every hard rule.

    If the current XI has a valid role composition, that exact composition is
    preserved. The best valid result keeps as many original players and
    captaincy choices as possible, then favours average fantasy points.
    """
    fixture = SQUADS.get(team.match)
    if not fixture:
        raise ValueError(f"No squad data available to repair {team.match}")

    pool_names = [
        name
        for squad in fixture["teams"].values()
        for name in squad
        if name in PLAYERS
    ]
    original_names = {pick.player for pick in team.players if pick.player in PLAYERS}
    original_reasons = {pick.player: pick.reason for pick in team.players}
    current_roles = Counter(PLAYERS[name]["role"] for name in original_names)
    role_order = tuple(RULES["composition"])

    if (
        len(team.players) == RULES["size"]
        and len(original_names) == RULES["size"]
        and all(
            low <= current_roles.get(role, 0) <= high
            for role, (low, high) in RULES["composition"].items()
        )
    ):
        target_roles = [tuple(current_roles[role] for role in role_order)]
    else:
        target_roles = [
            counts
            for counts in product(
                *(range(low, high + 1) for low, high in RULES["composition"].values())
            )
            if sum(counts) == RULES["size"]
        ]

    by_role = {
        role: [name for name in pool_names if PLAYERS[name]["role"] == role]
        for role in role_order
    }
    best = None

    def search(target, role_index, selected, credits, team_counts):
        nonlocal best
        if role_index == len(role_order):
            names = tuple(selected)
            captaincy_kept = int(team.captain in names) + int(team.vice_captain in names)
            retained = len(original_names.intersection(names))
            fantasy_points = sum(PLAYERS[name]["avg_fantasy_points"] for name in names)
            score = (captaincy_kept, retained, fantasy_points, -credits)
            if best is None or score > best[0]:
                best = (score, names)
            return

        role = role_order[role_index]
        count = target[role_index]
        for role_picks in combinations(by_role[role], count):
            next_credits = credits + sum(PLAYERS[name]["credits"] for name in role_picks)
            if next_credits > RULES["max_credits"]:
                continue
            next_teams = team_counts.copy()
            next_teams.update(PLAYERS[name]["team"] for name in role_picks)
            if any(value > RULES["max_per_team"] for value in next_teams.values()):
                continue
            search(target, role_index + 1, selected + list(role_picks), next_credits, next_teams)

    for target in target_roles:
        search(target, 0, [], 0.0, Counter())

    if best is None:
        raise ValueError(f"No valid XI can be constructed from the squad for {team.match}")

    names = list(best[1])
    ranked = sorted(names, key=lambda name: PLAYERS[name]["avg_fantasy_points"], reverse=True)
    captain = team.captain if team.captain in names else ranked[0]
    vice_captain = team.vice_captain if team.vice_captain in names and team.vice_captain != captain else next(
        name for name in ranked if name != captain
    )
    picks = [
        Pick(
            player=name,
            team=PLAYERS[name]["team"],
            role=PLAYERS[name]["role"],
            reason=original_reasons.get(name, f"Rule repair selection — {PLAYERS[name]['note']}"),
        )
        for name in names
    ]
    repaired = Team(
        match=team.match,
        captain=captain,
        vice_captain=vice_captain,
        players=picks,
        strategy=team.strategy,
    )
    issues, _ = validate_team(repaired)
    if issues:
        raise ValueError("Deterministic team repair failed validation: " + "; ".join(issues))
    return repaired


def player_pool_text(match: str | None = None) -> str:
    """A compact 'name — TEAM, ROLE, credits' pool for the correction prompt."""
    lines = [
        f"{n} — {p['team']}, {p['role']}, {p['credits']} cr"
        for n, p in PLAYERS.items()
    ]
    return "\n".join(lines)
