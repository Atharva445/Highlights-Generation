import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from parse_labels import extract_events

MATCH_PATH = "../SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United"

FIRST_HALF = os.path.join(MATCH_PATH, "1_720p.mkv")
SECOND_HALF = os.path.join(MATCH_PATH, "2_720p.mkv")

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# def get_structural_segments(video1, video2):

#     segments = []

#     # First half duration
#     dur1 = video1.duration

#     # Second half duration
#     dur2 = video2.duration

#     # 🔹 First half start
#     segments.append((0, 20))

#     # 🔹 First half end
#     segments.append((dur1 - 20, dur1))

#     # 🔹 Second half start (offset by 2700 sec)
#     segments.append((2700, 2700 + 20))

#     # 🔹 Full time (end of second half)
#     segments.append((2700 + dur2 - 20, 2700 + dur2))

#     return segments

def extract_highlights():

    label_file = os.path.join(MATCH_PATH, "Labels-v2.json")
    events = extract_events(label_file)

    for i, event in enumerate(events):

        timestamp = event["timestamp"]
        start = event["start"]
        end = event["end"]

        # Determine half
        if timestamp < 2700:
            video_path = FIRST_HALF
            clip_start = start
            clip_end = end
        else:
            video_path = SECOND_HALF
            clip_start = start - 2700
            clip_end = end - 2700

        print(f"Extracting clip {i+1} from {video_path}")

        with VideoFileClip(video_path) as video:
            highlight = video.subclipped(clip_start, clip_end)
            output_path = os.path.join(OUTPUT_DIR, f"highlight_{i+1}.mp4")
            highlight.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac"
            )

    print("All highlight clips extracted.")



if __name__ == "__main__":
    extract_highlights()