players = {
    "Virat Kohli": 75,
    "Rohit Sharma": 60
}

def get_player_stats(name):
    return players.get(name)

print(get_player_stats("Virat Kohli"))