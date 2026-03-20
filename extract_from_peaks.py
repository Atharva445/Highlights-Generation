import os
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips
from parse_labels import extract_events

MATCH_PATH = "../SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

FIRST_HALF = os.path.join(MATCH_PATH, "1_720p.mkv")
SECOND_HALF = os.path.join(MATCH_PATH, "2_720p.mkv")

OUTPUT_DIR = "../output"
SEGMENT_FILE = "peaks.npy"

PADDING = 1
MIN_DURATION = 5
MAX_CLIP_LENGTH = 15
MAX_SEGMENTS = 16
MAX_TOTAL_DURATION = 620  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

label_file = os.path.join(MATCH_PATH, "Labels-v2.json")
events = extract_events(label_file)
# ---------------------------------------------------
# 🔹 Merge overlapping segments
# ---------------------------------------------------
def merge_segments(segments):

    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]

    for curr in segments[1:]:
        prev = merged[-1]

        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)

    return merged

def get_structural_segments(video1, video2):

    dur1 = video1.duration
    dur2 = video2.duration

    return [
        (0, 20),
        (dur1 - 20, dur1),
        (2700, 2700 + 20),
        (2700 + dur2 - 20, 2700 + dur2)
    ]

# ---------------------------------------------------
# 🔹 Limit clip length
# ---------------------------------------------------
def cap_clip_length(segments):
    new_segments = []

    for s, e in segments:
        duration = e - s

        if duration > MAX_CLIP_LENGTH:
            # 🔴 Shift toward end (events usually near end)
            center = s + 0.7 * duration

            s_new = max(0, center - MAX_CLIP_LENGTH / 2)
            e_new = s_new + MAX_CLIP_LENGTH
        else:
            s_new, e_new = s, e

        new_segments.append((s_new, e_new))

    return new_segments

def segment_score(segment,events):
    s, e = segment
    duration = e - s

    event_weight = get_event_weight(segment, events)

    # Combine
    score = (0.5 * duration) + (2 * event_weight)

    return score

def get_event_weight(segment, events):
    s, e = segment

    for event in events:
        # Check overlap
        if not (e < event["start"] or s > event["end"]):
            return event["weight"]

    return 0.5  # default weight

goal_segments = [
    (e["timestamp"] - 3, e["timestamp"] + 8)
    for e in events if e["label"] == "Goal"
]
# ---------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------
def extract_clips():

    if not os.path.exists(SEGMENT_FILE):
        print("Segment file not found.")
        return

    segments = np.load(SEGMENT_FILE, allow_pickle=True)
    segments = [tuple(seg) for seg in segments]

    print(f"\nLoaded {len(segments)} raw segments")

    video1 = VideoFileClip(FIRST_HALF)
    video2 = VideoFileClip(SECOND_HALF)

    # 🔴 Add structural segments FIRST
    structural_segments = get_structural_segments(video1, video2)
    segments += structural_segments
    structural_set = set(structural_segments)

    # 🔴 Add goals FIRST (protect them)
    segments += goal_segments

    # 🔴 Merge
    segments = merge_segments(segments)

    # 🔴 Remove noise
    segments = [(s, e) for (s, e) in segments if (e - s) >= MIN_DURATION]

    # 🔴 Limit clip size
    segments = cap_clip_length(segments)

    # 🔴 Separate goals (VERY IMPORTANT)
    goal_set = set(goal_segments)

    # 🔴 Rank all
    ranked = sorted(segments, key=lambda x: segment_score(x, events), reverse=True)

    # 🔴 Always keep goals
    selected = []

    # 🔴 Always include goals
    for seg in ranked:
        if seg in goal_set:
            selected.append(seg)

    # 🔴 Always include structural segments
    for seg in ranked:
        if seg in structural_set and seg not in selected:
            selected.append(seg)

    # 🔴 Fill remaining slots
    for seg in ranked:
        if seg not in selected:
            selected.append(seg)
        if len(selected) >= MAX_SEGMENTS:
            break

    # 🔴 Final ordering
    segments = sorted(selected, key=lambda x: x[0])

    print(f"Using {len(segments)} final segments\n")

    clips = []
    total_duration = 0

    for i, (start, end) in enumerate(segments):

        duration = end - start

        if total_duration + duration > MAX_TOTAL_DURATION:
            break

        if start < 2700:
            video = video1
            clip_start = max(0, start - PADDING)
            clip_end = min(video.duration, end + PADDING)
        else:
            video = video2
            clip_start = max(0, start - 2700 - PADDING)
            clip_end = min(video.duration, end - 2700 + PADDING)

        print(f"[{i+1}] Extracting: {start:.2f}s → {end:.2f}s")

        clip = video.subclipped(clip_start, clip_end)
        clips.append(clip)

        total_duration += duration

    if clips:
        final_video = concatenate_videoclips(clips)
        final_output = os.path.join(OUTPUT_DIR, "final_highlights.mp4")

        final_video.write_videofile(
            final_output,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=4,
            logger=None
        )

        print(f"\nFinal highlights saved: {final_output}")

    video1.close()
    video2.close()

    print("\nHighlight extraction complete.")


if __name__ == "__main__":
    extract_clips()