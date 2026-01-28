import subprocess
import os
import sys
import pandas as pd
import ast

# --- CONFIGURATION ---
SOURCE_CSV_FILE = "panda70m.csv"
TARGET_VIDEOS = 450
OUTPUT_DIR = "../datasets/real"
OUTPUT_CSV = "real_captions.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for FFmpeg (Required for video trimming)
ffmpeg_exists = False
if os.path.exists("ffmpeg.exe"):
    ffmpeg_exists = True
elif subprocess.run("ffmpeg -version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
    ffmpeg_exists = True

if not ffmpeg_exists:
    print("CRITICAL ERROR: FFmpeg not found.")
    print("Please ensure ffmpeg.exe is in this directory.")
    # We continue, but downloads might fail if they are needed.

# Load Source CSV
if not os.path.exists(SOURCE_CSV_FILE):
    print(f"Error: Source file {SOURCE_CSV_FILE} not found.")
    sys.exit()

df = pd.read_csv(SOURCE_CSV_FILE)
print(f"Source CSV loaded: {len(df)} rows.")

# --- DOWNLOAD FUNCTION ---
def download_video_windows(yt_id, start, end, output_path):
    url = f"https://www.youtube.com/watch?v={yt_id}"
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "mp4[height<=360]",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--retries", "3",
        "-o", output_path,
        url
    ]
    
    try:
        # We capture output to check for specific YouTube errors
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            return True, "OK"
        else:
            return False, result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
    except Exception as e:
        return False, str(e)

# --- MAIN LOOP ---
print(f"\nStarting Dataset Verification & CSV Generation (Target: {TARGET_VIDEOS})...\n")

processed_count = 0
valid_data_list = []

# Iterate through the source CSV
for index, row in df.iterrows():
    if processed_count >= TARGET_VIDEOS:
        break

    yt_id = row['videoID']
    
    # Parse metadata
    try:
        timestamps = ast.literal_eval(row['timestamp'])
        captions = ast.literal_eval(row['caption'])
        
        if not timestamps or not captions: 
            continue
        
        start, end = timestamps[0]
        caption = captions[0]
    except:
        continue

    # Define the expected filename for this index
    video_filename = f"real_{processed_count:03d}.mp4"
    video_path = os.path.join(OUTPUT_DIR, video_filename)

    # CHECK 1: Does the file already exist?
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
        # The video exists locally. We assume it matches this row.
        print(f"[{processed_count}/{TARGET_VIDEOS}] File exists: {video_filename} -> Adding to CSV.")
        valid_data_list.append({'video_path': video_path, 'caption': caption})
        processed_count += 1
        continue

    # CHECK 2: If file is missing, we must download it to maintain alignment
    print(f"[{processed_count}/{TARGET_VIDEOS}] File missing. Downloading ID: {yt_id}...", end="")
    
    success, msg = download_video_windows(yt_id, start, end, video_path)
    
    if success and os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
        print(" OK")
        valid_data_list.append({'video_path': video_path, 'caption': caption})
        processed_count += 1
    else:
        # If download fails (e.g. video deleted), we SKIP this row completely.
        if "unavailable" in msg.lower() or "private" in msg.lower():
            print(" Video unavailable (Skipping)")
        else:
            print(f" Error: {msg[:30]}...")

# --- SAVE OUTPUT CSV ---
if valid_data_list:
    output_df = pd.DataFrame(valid_data_list)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCompleted. Generated {OUTPUT_CSV} with {len(valid_data_list)} entries.")
else:
    print("\nNo data found or generated.")