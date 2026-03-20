import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import os

MATCH_PATH = "../SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United"
VIDEO_PATHS = [
    os.path.join(MATCH_PATH, "1_720p.mkv"),
    os.path.join(MATCH_PATH, "2_720p.mkv")
]

MAX_SECONDS = 2400
RESIZE_DIM = (160, 90)
OUTPUT_SEGMENT_FILE = "peaks.npy"

MERGE_GAP_SECONDS = 20
MIN_SEGMENT_LENGTH = 3

MOTION_WEIGHT = 0.5
AUDIO_WEIGHT = 0.3
SHOT_WEIGHT = 0.2


# ---------------------------------------------------
# 1️⃣ Motion Detection (Optical Flow)
# ---------------------------------------------------
def compute_motion(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps)

    ret, prev_frame = cap.read()
    if not ret:
        return [], 0

    prev_frame = cv2.resize(prev_frame, RESIZE_DIM)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motion_scores = []
    shot_changes = []

    frame_count = 0
    max_frames = int(MAX_SECONDS * fps)

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > max_frames:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, RESIZE_DIM)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 🔹 Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_scores.append(np.mean(mag))

        # 🔹 Shot Change Detection
        hist1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        shot_changes.append(diff)

        prev_gray = gray

    cap.release()
    return np.array(motion_scores), np.array(shot_changes), 1


# ---------------------------------------------------
# 2️⃣ Audio Energy Detection
# ---------------------------------------------------
def compute_audio_energy(video_path):

    video = VideoFileClip(video_path)
    audio = video.audio.to_soundarray(fps=22050)
    video.close()

    audio_mono = np.mean(audio, axis=1)
    samples_per_second = 22050

    total_seconds = min(MAX_SECONDS, int(len(audio_mono) / samples_per_second))

    energy = []

    for sec in range(total_seconds):
        segment = audio_mono[sec * samples_per_second:(sec + 1) * samples_per_second]
        energy.append(np.mean(np.abs(segment)))

    return np.array(energy)


# ---------------------------------------------------
# 3️⃣ Normalize
# ---------------------------------------------------
def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)


# ---------------------------------------------------
# 4️⃣ Detect Highlights (Adaptive Threshold)
# ---------------------------------------------------
def detect_highlight_segments(fused_signal, fps):

    mean = np.mean(fused_signal)
    std = np.std(fused_signal)
    threshold = mean + 1.5 * std

    high_indices = np.where(fused_signal >= threshold)[0]

    if len(high_indices) == 0:
        return []

    segments = []
    start = high_indices[0]

    for i in range(1, len(high_indices)):
        if high_indices[i] != high_indices[i - 1] + 1:
            end = high_indices[i - 1]

            if (end - start) >= MIN_SEGMENT_LENGTH:
                segments.append((start, end))

            start = high_indices[i]

    segments.append((start, high_indices[-1]))

    # 🔹 Merge nearby segments
    merged = []
    prev_start, prev_end = segments[0]

    for curr_start, curr_end in segments[1:]:
        if (curr_start - prev_end) <= MERGE_GAP_SECONDS:
            prev_end = curr_end
        else:
            merged.append((prev_start, prev_end))
            prev_start, prev_end = curr_start, curr_end

    merged.append((prev_start, prev_end))

    highlight_segments = [(s / fps, e / fps) for s, e in merged]

    return highlight_segments


# ---------------------------------------------------
# 5️⃣ Goal Detection (Strong Audio Peaks)
# ---------------------------------------------------
def detect_goal_segments(audio_signal, fps):

    threshold = np.percentile(audio_signal, 97)
    goal_indices = np.where(audio_signal >= threshold)[0]

    goal_segments = []

    for idx in goal_indices:
        start = max(idx - 8, 0)
        end = idx + 15
        goal_segments.append((start / fps, end / fps))

    return goal_segments


# ---------------------------------------------------
# 6️⃣ Merge Overlapping Segments
# ---------------------------------------------------
def merge_segments(segments):

    if not segments:
        return []

    segments = sorted(segments)
    merged = [segments[0]]

    for curr in segments[1:]:
        prev = merged[-1]

        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)

    return merged


# ---------------------------------------------------
# 7️⃣ Add Structural Segments
# ---------------------------------------------------
def add_structural_segments(segments, video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()

    structural = [
        (0, min(20, duration)),
        (max(duration - 20, 0), duration)
    ]

    return merge_segments(segments + structural)


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
if __name__ == "__main__":

    all_segments = []
for idx, video_path in enumerate(VIDEO_PATHS):

    print(f"\nProcessing: {video_path}")

    motion_signal, shot_signal, fps = compute_motion(video_path)
    audio_signal = compute_audio_energy(video_path)

    min_len = min(len(motion_signal), len(audio_signal), len(shot_signal))

    motion_signal = motion_signal[:min_len]
    audio_signal = audio_signal[:min_len]
    shot_signal = shot_signal[:min_len]

    motion_norm = normalize(motion_signal)
    audio_norm = normalize(audio_signal)
    shot_norm = normalize(shot_signal)

    fused_signal = (
        0.5 * motion_norm +
        0.3 * audio_norm +
        0.2 * shot_norm
    )

    # 🔴 Stronger threshold (reduce noise)
    mean = np.mean(fused_signal)
    std = np.std(fused_signal)
    threshold = mean + 1.5 * std

    segments = detect_highlight_segments(fused_signal, fps)
    goal_segments = detect_goal_segments(audio_norm, fps)

    segments += goal_segments

    # 🔴 Offset second half
    if idx == 1:
        segments = [(s + 2700, e + 2700) for s, e in segments]

    all_segments.extend(segments)

# 🔴 Merge everything
all_segments = merge_segments(all_segments)

np.save(OUTPUT_SEGMENT_FILE, all_segments)

print(f"\nSaved {len(all_segments)} segments across full match.")