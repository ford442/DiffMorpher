import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from diffusers import DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.optimization import get_scheduler

# This is a helper function to encode prompts for SDXL's two text encoders
def encode_prompt_xl(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt):
    device = text_encoder.device
    
    # Tokenize
    tokenizers = [tokenizer, tokenizer_2]
    text_input_ids_list = []
    for t in tokenizers:
        text_input = t(
            prompt,
            padding="max_length",
            max_length=t.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_list.append(text_input.input_ids)
    
    text_input_ids = torch.cat(text_input_ids_list, dim=-1)

    # Encode
    prompt_embeds_list = []
    text_encoders = [text_encoder, text_encoder_2]
    for i, text_encoder_model in enumerate(text_encoders):
        prompt_embeds = text_encoder_model(text_input_ids_list[i].to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds

# This is a helper function for SDXL's conditioning
def get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype, device):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids


def train_lora_xl(
    image, prompt,
    unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
    lora_steps=200, lora_lr=2e-4, lora_rank=16,
    save_path=None, # CHANGED: We now take a single path for the save directory
):
    # --- Basic Setup ---
    set_seed(42)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
    )
    device = accelerator.device
    weight_dtype = torch.bfloat16
    
    # --- Freeze Models ---
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    
    lora_state_dict = None
    try:
        unet.train()
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank, init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        unet.add_adapter(lora_config)
        
        # --- Prepare Models for Training ---
        unet.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder_2.to(device, dtype=weight_dtype)

        # --- Optimizer ---
        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
        optimizer = torch.optim.AdamW(lora_layers, lr=lora_lr)
        
        # --- Prepare with Accelerator ---
        unet, optimizer = accelerator.prepare(unet, optimizer)

        # --- Prepare Data ---
        with torch.no_grad():
            prompt_embeds, pooled_embeds = encode_prompt_xl(
                text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt
            )
            add_time_ids = get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype, device=device)
        
        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
        
        image_transforms = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_image = image_transforms(image).unsqueeze(0).to(device, dtype=weight_dtype)

        with torch.no_grad():
            latents = vae.encode(train_image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise_scheduler = DDPMScheduler.from_pretrained('ford442/RealVisXL_V5.0_BF16', subfolder="scheduler")

        # --- Training Loop ---

        progress_bar = tqdm(range(lora_steps), desc=f"Training LoRA for {os.path.basename(save_path)}")
        for step in range(lora_steps):
            with accelerator.accumulate(unet):
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
            
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())

        # --- Save the LoRA ---
        unet = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unet)

        # Use the official diffusers save method
    finally:
        # This cleanup is still essential
        if "default" in unet.peft_config:
            unet.delete_adapters(["default"])
            print("Cleaned up temporary training adapter.")

    return lora_state_dict
