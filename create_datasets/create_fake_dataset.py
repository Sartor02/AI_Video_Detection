import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import pandas as pd
import os
import sys
import gc

from huggingface_hub import login

with open("../.env") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        os.environ[key] = value

MY_HF_TOKEN = os.environ["HF_TOKEN"]
if MY_HF_TOKEN:
    login(token=MY_HF_TOKEN)

# --- CONFIGURATION ---
CSV_FILE = "real_captions.csv"
# Save into the folder (used ModelScope)
OUTPUT_DIR = "datasets/fake_pyramid" 

# Lightweight standard Text-to-Video model
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"

# Generation settings
# ModelScope is native at 256x256.
# Is fine because the 3D-ViT detector will resize to 224x224 anyway.
HEIGHT = 256
WIDTH = 256   
NUM_FRAMES = 16  # About 2 seconds of video (standard for detection)
INFERENCE_STEPS = 25  # Good quality/speed trade-off

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Initializing pipeline ({MODEL_ID})...")

try:
    # Load pipeline in half precision (fp16)
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Optimized scheduler for better quality with fewer steps
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Memory optimizations
    # Offload model to CPU when GPU is not in use
    pipe.enable_model_cpu_offload()
    # Decode frames in chunks to avoid VRAM saturation
    pipe.enable_vae_slicing()

    print("Model loaded and optimized for 8GB VRAM.")

except Exception as e:
    print(f"\nERROR: {e}")
    print("Try running: pip install --upgrade diffusers transformers accelerate")
    sys.exit()

# --- LOAD DATA ---
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found. Run the real video download script first.")
    sys.exit()

df = pd.read_csv(CSV_FILE)
print(f"Found {len(df)} prompts to process.")

# --- GENERATION LOOP ---
print("\nStarting video generation...")

for index, row in df.iterrows():
    # File name aligned with the real one (fake_000 <-> real_000)
    video_filename = f"fake_{index:03d}.mp4"
    save_path = os.path.join(OUTPUT_DIR, video_filename)
    
    # Skip if already exists
    if os.path.exists(save_path):
        print(f"[{index}/{len(df)}] Already exists: {video_filename}")
        continue
        
    prompt = row['caption']
    
    # Prompt cleanup: ModelScope prefers simple English
    if not isinstance(prompt, str) or len(prompt) < 2:
        prompt = "A cinematic shot of a scene" 
        
    print(f"[{index}/{len(df)}] Generating: '{prompt[:40]}...'")
    
    try:
        # Generation
        video_frames = pipe(
            prompt,
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=INFERENCE_STEPS
        ).frames[0]
        
        # Save (8 fps = slow-motion style, typical for generators)
        export_to_video(video_frames, save_path, fps=8)
        
        # Force GPU memory cleanup after each video
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Saved: {video_filename}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        # If you run out of memory, try restarting the script
        continue

print(f"\nGeneration completed. Videos are in '{OUTPUT_DIR}'")
