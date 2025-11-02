from timeit import default_timer as timer
from datetime import timedelta
from PIL import Image
import os
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision import transforms
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from packaging import version
from PIL import Image
import tqdm

from transformers import AutoTokenizer, PretrainedConfig, CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

check_min_version("0.17.0")

# (import_model_class_from_model_name_or_path function remains the same)
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# (encode_prompt_xl function remains the same)
def encode_prompt_xl(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt):
    device = text_encoder.device
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]

    prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        prompt_embeds = text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )
        
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2] # Use penultimate layer
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

# (get_add_time_ids function remains the same)
def get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype, device):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids


#
# --- THIS IS THE FULLY CORRECTED FUNCTION ---
#
def train_lora(
    image, prompt, save_lora_dir, model_path=None, 
    text_encoder=None, text_encoder_2=None, 
    tokenizer=None, tokenizer_2=None, 
    vae=None, unet=None, noise_scheduler=None, 
    lora_steps=200, lora_lr=2e-4, lora_rank=16, 
    weight_name=None, safe_serialization=False, progress=tqdm
):
    
    accelerator = Accelerator(gradient_accumulation_steps=1)
    set_seed(0)

    # (Model loading logic remains the same)
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=None)
    if tokenizer_2 is None:
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", revision=None)
    if text_encoder is None:
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", revision=None)
    if text_encoder_2 is None:
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", revision=None)
    if vae is None:
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None)
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=None)
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    unet_dtype = unet.dtype 
    
    vae.to(device, dtype=unet_dtype)
    text_encoder.to(device)
    text_encoder_2.to(device)
    unet.to(device, dtype=unet_dtype) 

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # --- 1. Correctly instantiate all LoRA processors ---
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        # This is for SDXL CROSS-ATTENTION (is a torch.nn.Module)
        # Takes hidden_size, cross_attention_dim, and rank
        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, 
                cross_attention_dim=cross_attention_dim, 
                rank=lora_rank
            )
        
        # This is for SDXL SELF-ATTENTION (is a torch.nn.Module)
        # We FORCE LoRAAttnProcessor, which takes ONLY rank
        else:
            # This is for SDXL SELF-ATTENTION
            if hasattr(F, "scaled_dot_product_attention"):
                lora_attn_processor_class = LoRAAttnProcessor2_0
            else:
                lora_attn_processor_class = LoRAAttnProcessor

            # Initialize with NO arguments
            processor = lora_attn_processor_class() 

            # Set attributes *after* initialization
            processor.rank = lora_rank
            processor.cross_attention_dim = cross_attention_dim

            unet_lora_attn_procs[name] = processor
    
    unet.set_attn_processor(unet_lora_attn_procs)

    # --- 2. Correctly gather parameters ---
    # ALL processors are now Modules, so we just wrap them all
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)
    
    # Move the new module to the correct device and dtype
    unet_lora_layers.to(device, dtype=unet.dtype) 

    # Get parameters ONLY from this new module.
    params_to_optimize = list(unet_lora_layers.parameters())

    if not params_to_optimize:
            raise ValueError("No LoRA parameters found to optimize. "
                             "This is a critical error in the LoRA setup.")

    # --- 3. Optimizer creation ---
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # --- 4. Create the learning rate scheduler ---
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_steps,
    )

    # --- 5. Prepare with accelerate (ONLY the trainable module) ---
    unet_lora_layers, optimizer, lr_scheduler = accelerator.prepare(
        unet_lora_layers, optimizer, lr_scheduler
    )

    # --- 6. Get embeddings and conditioning ---
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt
        )
    
    add_time_ids = get_add_time_ids(
        (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype, device=device
    )
    
    bsz = 1 # Assuming batch size of 1 for LoRA training
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds.repeat(bsz, 1), "time_ids": add_time_ids.repeat(bsz, 1)}
    prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)

    if type(image) == np.ndarray:
        image = Image.fromarray(image)
        
    image_transforms = transforms.Compose(
        [
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image = image_transforms(image).to(device)
    image = image.unsqueeze(dim=0)
    
    with torch.no_grad():
        latents_dist = vae.encode(image.to(dtype=unet.dtype)).latent_dist

    # --- 7. Training loop ---
    for _ in progress.tqdm(range(lora_steps), desc="Training LoRA..."):
        model_input = latents_dist.sample() * vae.config.scaling_factor
        model_input = model_input.to(dtype=unet.dtype)

        noise = torch.randn_like(model_input)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        ).long()

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        model_pred = unet(
            noisy_model_input, 
            timesteps, 
            prompt_embeds, 
            added_cond_kwargs=added_cond_kwargs
        ).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # --- 8. Save weights ---
    unet_lora_layers = accelerator.unwrap_model(unet_lora_layers)
    
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
        weight_name=weight_name,
        safe_serialization=safe_serialization
    )

# (load_lora function remains the same)
def load_lora(unet, lora_0, lora_1, alpha):
    lora = {}
    for key in lora_0:
        lora[key] = (1 - alpha) * lora_0[key] + alpha * lora_1[key]
    unet.load_attn_procs(lora)
    return unet
