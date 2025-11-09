import torch
import os
import sys
import argparse
import logging
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.image as image
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib.text as text
import matplotlib.font_manager as font_manager
import matplotlib.colors as colors

current_dir = os.path.dirname(os.path.abspath(__file__))

# pretrained model path
# INPAINTING_MODEL_PATH = "/home/liying/Documents/stable-diffusion-inpainting"
INPAINTING_MODEL_PATH = "/home/liying/Desktop/stable-diffusion-inpainting"
INPAINTING_2_MODEL_PATH = "/home/liying/Desktop/stable-diffusion-2-inpainting"

STABLE_DIFFUSION_V1_5_MODEL_PATH = "/home/liying/Documents/stable-diffusion-v1-5"

# LISA model path
LISA_7B_MODEL_PATH = "/home/liying/Documents/LISA-7B-v1"
LISA_13B_MODEL_PATH = "/home/liying/Documents/LISA-13B-llama2-v1"


# dataset path
CHECKPOINTS_DIR = os.path.join(current_dir, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

INPUT_DIR = os.path.join(current_dir, "input")
os.makedirs(INPUT_DIR, exist_ok=True)

INPUT_IMAGES_JSON_FILE = os.path.join(INPUT_DIR, "images_1.json")
if not os.path.exists(INPUT_IMAGES_JSON_FILE):
    raise FileNotFoundError(f"Input images JSON file not found at {INPUT_IMAGES_JSON_FILE}")

INPUT_IMAGES_2_JSON_FILE = os.path.join(INPUT_DIR, "images_2.json")
if not os.path.exists(INPUT_IMAGES_2_JSON_FILE):
    raise FileNotFoundError(f"Input images JSON file not found at {INPUT_IMAGES_2_JSON_FILE}")

# result path
OUTPUT_DIR = os.path.join(current_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vis_new")
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

EDITING_RESULTS_DIR = os.path.join(OUTPUT_DIR, "editing_results")
os.makedirs(EDITING_RESULTS_DIR, exist_ok=True)
