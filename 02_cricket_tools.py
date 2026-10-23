players = {
    "MS Dhoni": 100,
    "Virat Kohli": 75,
    "Rohit Sharma": 60
}

def get_player_stats(name):
    return players.get(name)

print(get_player_stats("Virat Kohli"))

tool_schema = {
    "type": "function",
    "function": {
        "name": "get_player_stats",
        "description": "Get a player's sample score",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    }
}

def get_pitch_report():
    return "Spin-friendly pitch"

print(get_pitch_report())