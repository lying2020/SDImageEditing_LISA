import argparse
import os
# Fix Qt platform plugin issue for OpenCV
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Auto-fix CUDA library path for bitsandbytes
torch_lib = os.path.expanduser("~/.local/lib/python3.10/site-packages/torch/lib")
if os.path.exists(torch_lib):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if torch_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{current_ld_path}"
import sys
import warnings
# Set environment to suppress bitsandbytes warnings
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
# CRITICAL: Set environment variables BEFORE any other imports
# This prevents bitsandbytes CUDA setup errors during import
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*libcudart.*")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
# This must be done BEFORE importing transformers (which imports accelerate, which imports bitsandbytes)
from transformers import AutoTokenizer, CLIPImageProcessor
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

from utils.utils import EditingJsonDataset
import project
from project import LISA_7B_MODEL_PATH, LISA_13B_MODEL_PATH, VIS_OUTPUT_DIR, INPUT_IMAGES_JSON_FILE, INPUT_IMAGES_2_JSON_FILE


def parse_args():
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default=LISA_13B_MODEL_PATH)
    parser.add_argument("--input_images_json_file", default=INPUT_IMAGES_2_JSON_FILE, help="JSON file to load input images")
    parser.add_argument("--vis_save_path", default=VIS_OUTPUT_DIR, help="Directory to save visualization results")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"], help="precision for inference")
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    args = parser.parse_args()
    return args


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def chat(args, model, clip_image_processor, transform, tokenizer, image_path, prompt):
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    if not os.path.exists(image_path):
        print("File not found in {}".format(image_path))
        return

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list, max_new_tokens=512, tokenizer=tokenizer)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    print("text_output: ", text_output)

    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        save_path_mask = "{}/{}_mask_{}.jpg".format(
            args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
        )
        cv2.imwrite(save_path_mask, pred_mask * 100)
        print("{} has been saved.".format(save_path_mask))

        save_path_masked_img = "{}/{}_masked_img_{}.jpg".format(
            args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
        )
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_masked_img, save_img)
        print("{} has been saved.".format(save_path_masked_img))

    return text_output, save_path_mask, save_path_masked_img

def main(args):
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(args.version, cache_dir=None, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        # Lazy import BitsAndBytesConfig to avoid bitsandbytes CUDA setup issues
        from transformers import BitsAndBytesConfig
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        # Lazy import BitsAndBytesConfig to avoid bitsandbytes CUDA setup issues
        from transformers import BitsAndBytesConfig
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
        # Debug: Check if model is on GPU
        print("=" * 60)
        print("Model Device Check")
        print("=" * 60)
        try:
            model_device = next(model.parameters()).device
            print(f"Model device: {model_device}")
            if model_device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"✅ Model is on GPU!")
                print(f"GPU memory used: {gpu_mem:.2f} GB")
            else:
                print(f"❌ WARNING: Model is on {model_device}, not GPU!")
                print("This will be VERY slow. Check CUDA setup.")
        except Exception as e:
            print(f"Error checking model device: {e}")
        print("=" * 60)
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        model_engine = deepspeed.init_inference(model=model, dtype=torch.half, replace_with_kernel_inject=True, replace_method="auto")
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    # Debug: Check if model is on GPU
    print("=" * 60)
    print("Model Device Check")
    print("=" * 60)
    try:
        model_device = next(model.parameters()).device
        print(f"Model device: {model_device}")
        if model_device.type == "cuda":
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"✅ Model is on GPU!")
            print(f"GPU memory used: {gpu_mem:.2f} GB")
        else:
            print(f"❌ WARNING: Model is on {model_device}, not GPU!")
            print("This will be VERY slow. Check CUDA setup.")
    except Exception as e:
        print(f"Error checking model device: {e}")

    print("=" * 60)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    # Set required args for EditingJsonDataset
    dataset = EditingJsonDataset(args.input_images_json_file)
    for idx, (image, original_prompt, editing_prompt) in enumerate(dataset):
        # Get image path from dataset
        image_path = os.path.join(dataset.image_dir, dataset.image_files[idx])
        text_output, save_path_mask, save_path_masked_img = chat(args, model, clip_image_processor, transform, tokenizer, image_path, original_prompt)
        print(f"\n[{idx+1}/{len(dataset)}] text_output: {text_output}")
        print(f"[{idx+1}/{len(dataset)}] save_path_mask: {save_path_mask}")
        print(f"[{idx+1}/{len(dataset)}] save_path_masked_img: {save_path_masked_img}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
