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

# --- CONFIGURAZIONE ---
CSV_FILE = "real_captions.csv"
# Salviamo nella cartella che il dataloader si aspetta, anche se usiamo ModelScope
OUTPUT_DIR = "datasets/fake_pyramid" 

# Modello Text-to-Video standard leggero (funziona su 8GB VRAM)
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"

# Impostazioni Generazione
# ModelScope è nativo a 256x256. 
# È perfetto perché il tuo detector 3D-ViT ridimensiona comunque a 224x224.
HEIGHT = 256
WIDTH = 256   
NUM_FRAMES = 16  # Circa 2 secondi di video (standard per detection)
INFERENCE_STEPS = 25 # Buon compromesso qualità/velocità

# --- SETUP ---
# Installazione automatica librerie se mancano
try:
    import diffusers
except ImportError:
    print("Installazione librerie necessarie...")
    os.system(f"{sys.executable} -m pip install diffusers transformers accelerate")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Inizializzazione Pipeline ({MODEL_ID})...")

try:
    # 1. Caricamento Pipeline in mezza precisione (fp16)
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Scheduler ottimizzato (DPM) per qualità migliore con pochi step
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # 2. Ottimizzazioni Memoria (CRUCIALI per la tua 4060)
    # Sposta il modello sulla CPU quando non serve la GPU
    pipe.enable_model_cpu_offload()
    # Decodifica i frame a pezzi per non saturare la VRAM
    pipe.enable_vae_slicing()

    print("✅ Modello caricato e ottimizzato per 8GB VRAM.")

except Exception as e:
    print(f"\n❌ ERRORE: {e}")
    print("Prova a eseguire: pip install --upgrade diffusers transformers accelerate")
    sys.exit()

# --- CARICAMENTO DATI ---
if not os.path.exists(CSV_FILE):
    print(f"❌ Errore: {CSV_FILE} non trovato. Esegui prima lo script di download dei video REALI.")
    sys.exit()

df = pd.read_csv(CSV_FILE)
print(f"Trovati {len(df)} prompt da elaborare.")

# --- LOOP GENERAZIONE ---
print("\nInizio generazione video...")

for index, row in df.iterrows():
    # Nome file allineato con quello reale (fake_000 <-> real_000)
    video_filename = f"fake_{index:03d}.mp4"
    save_path = os.path.join(OUTPUT_DIR, video_filename)
    
    # Salta se esiste già
    if os.path.exists(save_path):
        print(f"[{index}/{len(df)}] Esiste già: {video_filename}")
        continue
        
    prompt = row['caption']
    
    # Pulizia prompt: ModelScope preferisce inglese semplice
    if not isinstance(prompt, str) or len(prompt) < 2:
        prompt = "A cinematic shot of a scene" 
        
    print(f"[{index}/{len(df)}] Generando: '{prompt[:40]}...'")
    
    try:
        # Generazione
        video_frames = pipe(
            prompt,
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=INFERENCE_STEPS
        ).frames[0]
        
        # Salvataggio (8 fps = video rallentato stile slow-mo, tipico dei generatori)
        export_to_video(video_frames, save_path, fps=8)
        
        # Pulizia forzata della memoria GPU dopo ogni video
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"   ✅ Salvato: {video_filename}")
        
    except Exception as e:
        print(f"   ❌ ERRORE: {e}")
        # Se finisci la memoria, prova a riavviare lo script
        continue

print(f"\n🎉 Generazione Completata! I video sono in '{OUTPUT_DIR}'")
