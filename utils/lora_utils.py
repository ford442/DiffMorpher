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
import tqdm
from peft import LoraConfig

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

def get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype, device):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids

def train_lora(
  image, prompt, save_lora_dir, model_path=None,
  text_encoder=None, text_encoder_2=None,
  tokenizer=None, tokenizer_2=None,
  vae=None, unet=None,
  lora_steps=200, lora_lr=2e-4, lora_rank=16,
  weight_name=None, safe_serialization=False, progress=tqdm
):
  accelerator = Accelerator(gradient_accumulation_steps=1)
  set_seed(0)
  weight_dtype = torch.bfloat16
  if tokenizer is None:
      tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=None)
  if tokenizer_2 is None:
      tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", revision=None)
  if text_encoder is None:
      text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", revision=None, torch_dtype=weight_dtype)
  if text_encoder_2 is None:
      text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", revision=None, torch_dtype=weight_dtype)
  if vae is None:
      vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None, torch_dtype=weight_dtype)
  #if unet is None:
  unet = UNet2DConditionModel.from_pretrained('ford442/RealVisXL_V5.0_BF16', subfolder="unet", revision=None, torch_dtype=weight_dtype)
  #if noise_scheduler is None:
  noise_scheduler = DDPMScheduler.from_pretrained('ford442/RealVisXL_V5.0_BF16', subfolder="scheduler")
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  unet_dtype = weight_dtype
  vae.to(weight_dtype)
  text_encoder.to(device)
  text_encoder_2.to(device)
  unet.to(device, weight_dtype)
  # 1. Freeze all parameters in the UNet before adding the adapter
  vae.requires_grad_(False)
  text_encoder.requires_grad_(False)
  text_encoder_2.requires_grad_(False)
  unet.requires_grad_(False) # Freeze the entire UNet first
  # 2. Use the modern 'add_adapter' method for LoRA
  # This replaces the manual loop and 'set_attn_processor'
  lora_config = LoraConfig(
      r=lora_rank,
      lora_alpha=lora_rank, # Often set to the same as rank
      # This targets the specific layers needed for SDXL attention blocks
      target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
  )
  unet.add_adapter(lora_config)
  # 3. Gather the trainable LoRA parameters (this part remains conceptually the same)
  # The '.parameters()' method will now correctly include the new adapter weights.
  params_to_optimize = []
  for name, param in unet.named_parameters():
      if param.requires_grad: # The only params with requires_grad=True are the LoRA ones
          params_to_optimize.append(param)
  if not params_to_optimize:
      raise ValueError("No trainable parameters found. The LoRA adapter may not have been added correctly.")
  # 4. Optimizer creation (remains the same)
  optimizer = torch.optim.AdamW(
      params_to_optimize,
      lr=lora_lr,
      betas=(0.9, 0.999),
      weight_decay=1e-2,
      eps=1e-08,
  )
  # 5. LR scheduler creation (remains the same)
  lr_scheduler = get_scheduler(
      "constant",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=lora_steps,
  )
  # 6. Prepare with accelerate (remains the same)
  # Note: we are preparing the entire unet, which now contains the LoRA adapter
  unet, optimizer, lr_scheduler = accelerator.prepare(
      unet, optimizer, lr_scheduler
  )
  # 7. Get embeddings and conditioning (remains the same)
  with torch.no_grad():
      prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(
          text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt
      )
  add_time_ids = get_add_time_ids(
      (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype, device=device
  )
  bsz = 1
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
      latents_dist = vae.encode(image.to(dtype=weight_dtype)).latent_dist
  # 8. Set unet to train mode
  unet.train()
  # Training loop (remains the same)
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

  # 9. Save weights using the modern method
  unet = accelerator.unwrap_model(unet)
  # This is the new, correct way to save LoRA weights from a model
  # that has an adapter attached.
  unet.save_lora_weights(
      save_directory=save_lora_dir,
      weight_name=weight_name,
      safe_serialization=safe_serialization
  )

def load_lora(unet, lora_0, lora_1, alpha):
    lora = {}
    for key in lora_0:
        lora[key] = (1 - alpha) * lora_0[key] + alpha * lora_1[key]
    unet.load_attn_procs(lora)
    return unet
