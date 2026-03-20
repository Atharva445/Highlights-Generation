def game_time_to_seconds(game_time):
    """
    Converts '1 - 35:42' into total seconds from start of match.
    """

    half, time = game_time.split(" - ")
    minutes, seconds = time.split(":")

    total_seconds = int(minutes) * 60 + int(seconds)

    # Add 45 minutes offset for second half
    if half.strip() == "2":
        total_seconds += 45 * 60

    return total_seconds