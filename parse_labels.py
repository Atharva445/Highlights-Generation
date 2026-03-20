import json
import os
from utils import game_time_to_seconds
from config import EVENTS_OF_INTEREST, PRE_EVENT_SECONDS, POST_EVENT_SECONDS

# 🔹 Event importance weights (used for ranking later)
EVENT_WEIGHTS = {
    "Goal": 1.0,
    "Red card": 0.95,
    "Yellow card": 0.8,
    "Shots on target": 0.9,
    "Shots off target": 0.75,
    "Substitution": 0.7,
    "Corner": 0.65,
    "Foul": 0.6,
    "Offside": 0.55
}

# 🔹 Minimum weight filter (optional tuning)
MIN_EVENT_WEIGHT = 0.6


def extract_events(label_path):
    if not os.path.exists(label_path):
        print("Label file not found.")
        return []

    with open(label_path, "r") as f:
        data = json.load(f)

    events = []

    for annotation in data.get("annotations", []):
        label = annotation.get("label", "")

        # ✅ Filter only relevant events
        if label not in EVENTS_OF_INTEREST:
            continue

        # ✅ Convert time safely
        try:
            game_time = annotation["gameTime"]
            seconds = game_time_to_seconds(game_time)
        except Exception:
            continue

        # ✅ Compute clip window
        start_time = max(0, seconds - PRE_EVENT_SECONDS)
        end_time = seconds + POST_EVENT_SECONDS

        # ✅ Assign importance
        weight = EVENT_WEIGHTS.get(label, 0.5)

        # ✅ Optional filtering (remove very weak events)
        if weight < MIN_EVENT_WEIGHT:
            continue

        events.append({
            "label": label,
            "timestamp": seconds,
            "start": start_time,
            "end": end_time,
            "weight": weight
        })

    # ✅ Sort by importance first, then time
    events = sorted(events, key=lambda x: (-x["weight"], x["timestamp"]))

    print(f"\nExtracted {len(events)} important events.")

    return events


if __name__ == "__main__":

    label_file = "../SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United/Labels-v2.json"

    events = extract_events(label_file)

    for e in events:
        print(f"{e['label']} | {e['timestamp']}s | weight={e['weight']}")