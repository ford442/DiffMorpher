import os
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import safetensors
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import StableDiffusionXLPipeline # SDXL Change
from argparse import ArgumentParser
import inspect

# SDXL Change: Import from our new xl utils
from utils.model_utils_xl import get_img, slerp, do_replace_attn 
# SDXL Change: Import from original lora_utils (will be updated to lora_utils_xl later)
from utils.lora_utils import train_lora, load_lora
from utils.alpha_scheduler import AlphaScheduler

# StoreProcessor and LoadProcessor are identical to the original model.py
# (They are model-agnostic as they only manipulate self-attention)
class StoreProcessor():
    def __init__(self, original_processor, value_dict, name):
        self.original_processor = original_processor
        self.value_dict = value_dict
        self.name = name
        self.value_dict[self.name] = dict()
        self.id = 0

    def __call__(self, attn, hidden_states, *args, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            self.value_dict[self.name][self.id] = hidden_states.detach()
            self.id += 1
        res = self.original_processor(attn, hidden_states, *args,
                                      encoder_hidden_states=encoder_hidden_states,
                                      attention_mask=attention_mask,
                                      **kwargs)
        return res

class LoadProcessor():
    def __init__(self, original_processor, name, img0_dict, img1_dict, alpha, beta=0, lamd=0.6):
        super().__init__()
        self.original_processor = original_processor
        self.name = name
        self.img0_dict = img0_dict
        self.img1_dict = img1_dict
        self.alpha = alpha
        self.beta = beta
        self.lamd = lamd
        self.id = 0

    def __call__(self, attn, hidden_states, *args, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            if self.id < 50 * self.lamd:
                map0 = self.img0_dict[self.name][self.id]
                map1 = self.img1_dict[self.name][self.id]
                cross_map = self.beta * hidden_states + \
                    (1 - self.beta) * ((1 - self.alpha) * map0 + self.alpha * map1)
                res = self.original_processor(attn, hidden_states, *args,
                                              encoder_hidden_states=cross_map,
                                              attention_mask=attention_mask,
                                              **kwargs)
            else:
                res = self.original_processor(attn, hidden_states, *args,
                                              encoder_hidden_states=encoder_hidden_states,
                                              attention_mask=attention_mask,
                                              **kwargs)
            self.id += 1
            if self.id == len(self.img0_dict[self.name]):
                self.id = 0
        else:
            res = self.original_processor(attn, hidden_states, *args,
                                          encoder_hidden_states=encoder_hidden_states,
                                          attention_mask=attention_mask,
                                          **kwargs)
        return res


# SDXL Change: Inherit from StableDiffusionXLPipeline
class DiffMorpherPipelineXL(StableDiffusionXLPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection, # SDXL Change
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer, # SDXL Change
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder=None,
        requires_safety_checker: bool = True,
    ):
        # SDXL Change: Simplified __init__ to pass all components to the SDXL parent
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.img0_dict = dict()
        self.img1_dict = dict()

    # inv_step, image2latent, latent2image, latent2image_grad, step
    # are all identical to the original model.py and can be copied over verbatim.
    # ... (Copy inv_step from model.py) ...
    # ... (Copy image2latent from model.py) ...
    # ... (Copy latent2image from model.py) ...
    # ... (Copy latent2image_grad from model.py) ...
    # ... (Copy step from model.py) ...
    
    # SDXL Change: ddim_inversion needs to be updated for dual encoders
    @torch.no_grad()
    def ddim_inversion(self, latent, prompt_embeds, pooled_prompt_embeds):
        timesteps = reversed(self.scheduler.timesteps)
        
        # SDXL Change: Prepare added_cond_kwargs
        # We assume default resolution 1024x1024, no cropping
        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype
        ).to(self.device)
        
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm.tqdm(timesteps, desc="DDIM inversion")):
                
                # SDXL Change: U-Net call now includes encoder_hidden_states and added_cond_kwargs
                eps = self.unet(
                    latent, 
                    t, 
                    encoder_hidden_states=prompt_embeds, 
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps

        return latent

    # SDXL Change: cal_latent needs to interpolate both sets of embeddings
    @torch.no_grad()
    def cal_latent(self, num_inference_steps, guidance_scale, unconditioning, 
                   img_noise_0, img_noise_1, 
                   prompt_embeds_0, pooled_embeds_0,  # SDXL Change
                   prompt_embeds_1, pooled_embeds_1,  # SDXL Change
                   lora_0, lora_1, alpha, use_lora, fix_lora=None):
        
        latents = slerp(img_noise_0, img_noise_1, alpha, self.use_adain)
        
        # SDXL Change: Interpolate both prompt and pooled embeddings
        prompt_embeds = (1 - alpha) * prompt_embeds_0 + alpha * prompt_embeds_1
        pooled_embeds = (1 - alpha) * pooled_embeds_0 + alpha * pooled_embeds_1

        # SDXL Change: Prepare added_cond_kwargs
        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype
        ).to(self.device)
        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

        # Handle CFG for pooled embeds
        if guidance_scale > 1.:
            # Unconditional embeds are the first half
            neg_pooled_embeds = pooled_embeds[:pooled_embeds.shape[0]//2]
            pooled_embeds = pooled_embeds[pooled_embeds.shape[0]//2:]
            # Duplicate pooled embeds for CFG
            cfg_pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds], dim=0)
            added_cond_kwargs["text_embeds"] = cfg_pooled_embeds


        self.scheduler.set_timesteps(num_inference_steps)
        if use_lora:
            if fix_lora is not None:
                self.unet = load_lora(self.unet, lora_0, lora_1, fix_lora)
            else:
                self.unet = load_lora(self.unet, lora_0, lora_1, alpha)

        for i, t in enumerate(tqdm.tqdm(self.scheduler.timesteps, desc=f"DDIM Sampler, alpha={alpha}")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            
            # Note: unconditioning logic from original model.py is omitted for simplicity
            # It would need to be adapted for dual embeds if required

            # SDXL Change: U-Net call with new kwargs
            noise_pred = self.unet(
                model_inputs, 
                t, 
                encoder_hidden_states=prompt_embeds, 
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    # SDXL Change: New function to get dual embeddings
    @torch.no_grad()
    def get_text_embeddings(self, prompt, guidance_scale, neg_prompt, batch_size):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else self.device
        
        if neg_prompt is None:
            neg_prompt = ""

        # Use the pipeline's internal encoding function
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=[prompt] * batch_size,
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=[neg_prompt] * batch_size,
        )

        if guidance_scale > 1.:
            # Concatenate for CFG
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            pooled_embeds = pooled_prompt_embeds

        return prompt_embeds, pooled_embeds # Return two sets of embeddings

    # SDXL Change: Main pipeline call
    def __call__(
            self,
            img_0=None,
            img_1=None,
            img_path_0=None,
            img_path_1=None,
            prompt_0="",
            prompt_1="",
            save_lora_dir="./lora",
            load_lora_path_0=None,
            load_lora_path_1=None,
            lora_steps=200,
            lora_lr=2e-4,
            lora_rank=16,
            batch_size=1,
            height=1024, # SDXL Change
            width=1024,  # SDXL Change
            num_inference_steps=50,
            num_actual_inference_steps=None,
            guidance_scale=7.5, # SDXL Change: Use a more standard CFG
            attn_beta=0,
            lamd=0.6,
            use_lora=True,
            use_adain=True,
            use_reschedule=True,
            output_path="./results",
            num_frames=50,
            fix_lora=None,
            progress=tqdm,
            unconditioning=None,
            neg_prompt=None,
            save_intermediates=False,
            **kwds):

        self.scheduler.set_timesteps(num_inference_steps)
        self.use_lora = use_lora
        self.use_adain = use_adain
        self.use_reschedule = use_reschedule
        self.output_path = output_path

        if img_0 is None:
            img_0 = Image.open(img_path_0).convert("RGB")
        if img_1 is None:
            img_1 = Image.open(img_path_1).convert("RGB")

        if self.use_lora:
            print("Loading lora...")
            # This logic remains the same, but it will call the (new) lora_utils_xl.py
            # which needs to be updated to train for SDXL
            if not load_lora_path_0:
                weight_name = f"{output_path.split('/')[-1]}_lora_0_xl.ckpt" # SDXL Change
                load_lora_path_0 = save_lora_dir + "/" + weight_name
                if not os.path.exists(load_lora_path_0):
                    train_lora(img_0, prompt_0, save_lora_dir, self.text_encoder, self.text_encoder_2, 
                               self.tokenizer, self.tokenizer_2, self.vae, self.unet, self.scheduler, 
                               lora_steps, lora_lr, lora_rank, weight_name=weight_name)
            
            lora_0 = torch.load(load_lora_path_0, map_location="cpu")

            if not load_lora_path_1:
                weight_name = f"{output_path.split('/')[-1]}_lora_1_xl.ckpt" # SDXL Change
                load_lora_path_1 = save_lora_dir + "/" + weight_name
                if not os.path.exists(load_lora_path_1):
                    train_lora(img_1, prompt_1, save_lora_dir, self.text_encoder, self.text_encoder_2, 
                               self.tokenizer, self.tokenizer_2, self.vae, self.unet, self.scheduler, 
                               lora_steps, lora_lr, lora_rank, weight_name=weight_name)

            lora_1 = torch.load(load_lora_path_1, map_location="cpu")
        else:
            lora_0 = lora_1 = None

        # SDXL Change: Get both sets of embeddings
        prompt_embeds_0, pooled_embeds_0 = self.get_text_embeddings(
            prompt_0, guidance_scale, neg_prompt, batch_size)
        prompt_embeds_1, pooled_embeds_1 = self.get_text_embeddings(
            prompt_1, guidance_scale, neg_prompt, batch_size)
        
        img_0 = get_img(img_0) # Uses get_img from model_utils_xl (1024)
        img_1 = get_img(img_1) # Uses get_img from model_utils_xl (1024)
        
        if self.use_lora:
            self.unet = load_lora(self.unet, lora_0, lora_1, 0)
        img_noise_0 = self.ddim_inversion(
            self.image2latent(img_0), prompt_embeds_0, pooled_embeds_0) # SDXL Change
        
        if self.use_lora:
            self.unet = load_lora(self.unet, lora_0, lora_1, 1)
        img_noise_1 = self.ddim_inversion(
            self.image2latent(img_1), prompt_embeds_1, pooled_embeds_1) # SDXL Change

        print("latents shape: ", img_noise_0.shape)
        
        original_processor = list(self.unet.attn_processors.values())[0]
        
        # This morph function is adapted from the original model.py
        def morph(alpha_list, progress, desc):
            images = []
            if attn_beta is not None:
                if self.use_lora:
                    self.unet = load_lora(
                        self.unet, lora_0, lora_1, 0 if fix_lora is None else fix_lora)

                # (Set up StoreProcessor... identical to original)
                attn_processor_dict = {}
                for k in self.unet.attn_processors.keys():
                    if do_replace_attn(k):
                        if self.use_lora:
                            attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k],
                                                                    self.img0_dict, k)
                        else:
                            attn_processor_dict[k] = StoreProcessor(original_processor,
                                                                    self.img0_dict, k)
                    else:
                        attn_processor_dict[k] = self.unet.attn_processors[k]
                self.unet.set_attn_processor(attn_processor_dict)

                # SDXL Change: Pass dual embeds to cal_latent
                latents = self.cal_latent(
                    num_inference_steps, guidance_scale, unconditioning,
                    img_noise_0, img_noise_1,
                    prompt_embeds_0, pooled_embeds_0,
                    prompt_embeds_1, pooled_embeds_1,
                    lora_0, lora_1, alpha_list[0], False, fix_lora
                )
                first_image = self.latent2image(latents)
                first_image = Image.fromarray(first_image)
                if save_intermediates:
                    first_image.save(f"{self.output_path}/{0:02d}.png")

                # (Set up StoreProcessor for img1... identical to original)
                if self.use_lora:
                    self.unet = load_lora(
                        self.unet, lora_0, lora_1, 1 if fix_lora is None else fix_lora)
                attn_processor_dict = {}
                for k in self.unet.attn_processors.keys():
                    if do_replace_attn(k):
                        if self.use_lora:
                            attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k],
                                                                    self.img1_dict, k)
                        else:
                            attn_processor_dict[k] = StoreProcessor(original_processor,
                                                                    self.img1_dict, k)
                    else:
                        attn_processor_dict[k] = self.unet.attn_processors[k]
                self.unet.set_attn_processor(attn_processor_dict)

                # SDXL Change: Pass dual embeds to cal_latent
                latents = self.cal_latent(
                    num_inference_steps, guidance_scale, unconditioning,
                    img_noise_0, img_noise_1,
                    prompt_embeds_0, pooled_embeds_0,
                    prompt_embeds_1, pooled_embeds_1,
                    lora_0, lora_1, alpha_list[-1], False, fix_lora
                )
                last_image = self.latent2image(latents)
                last_image = Image.fromarray(last_image)
                if save_intermediates:
                    last_image.save(f"{self.output_path}/{num_frames - 1:02d}.png")

                # Main loop
                for i in progress.tqdm(range(1, num_frames - 1), desc=desc):
                    alpha = alpha_list[i]
                    if self.use_lora:
                        self.unet = load_lora(
                            self.unet, lora_0, lora_1, alpha if fix_lora is None else fix_lora)
                    
                    # (Set up LoadProcessor... identical to original)
                    attn_processor_dict = {}
                    for k in self.unet.attn_processors.keys():
                        if do_replace_attn(k):
                            if self.use_lora:
                                attn_processor_dict[k] = LoadProcessor(
                                    self.unet.attn_processors[k], k, self.img0_dict, self.img1_dict, alpha, attn_beta, lamd)
                            else:
                                attn_processor_dict[k] = LoadProcessor(
                                    original_processor, k, self.img0_dict, self.img1_dict, alpha, attn_beta, lamd)
                        else:
                            attn_processor_dict[k] = self.unet.attn_processors[k]
                    self.unet.set_attn_processor(attn_processor_dict)

                    # SDXL Change: Pass dual embeds to cal_latent
                    latents = self.cal_latent(
                        num_inference_steps, guidance_scale, unconditioning,
                        img_noise_0, img_noise_1,
                        prompt_embeds_0, pooled_embeds_0,
                        prompt_embeds_1, pooled_embeds_1,
                        lora_0, lora_1, alpha_list[i], False, fix_lora
                    )
                    image = self.latent2image(latents)
                    image = Image.fromarray(image)
                    if save_intermediates:
                        image.save(f"{self.output_path}/{i:02d}.png")
                    images.append(image)
                images = [first_image] + images + [last_image]
            
            # (This 'else' block for no attn_beta is identical, but calls the modified cal_latent)
            else:
                for k, alpha in enumerate(alpha_list):
                    latents = self.cal_latent(
                        num_inference_steps, guidance_scale, unconditioning,
                        img_noise_0, img_noise_1,
                        prompt_embeds_0, pooled_embeds_0,
                        prompt_embeds_1, pooled_embeds_1,
                        lora_0, lora_1, alpha_list[k], self.use_lora, fix_lora
                    )
                    image = self.latent2image(latents)
                    image = Image.fromarray(image)
                    if save_intermediates:
                        image.save(f"{self.output_path}/{k:02d}.png")
                    images.append(image)

            return images
        
        # (Reschedule logic is identical)
        with torch.no_grad():
            if self.use_reschedule:
                alpha_scheduler = AlphaScheduler()
                alpha_list = list(torch.linspace(0, 1, num_frames))
                images_pt = morph(alpha_list, progress, "Sampling...")
                images_pt = [transforms.ToTensor()(img).unsqueeze(0)
                             for img in images_pt]
                alpha_scheduler.from_imgs(images_pt)
                alpha_list = alpha_scheduler.get_list()
                print(alpha_list)
                images = morph(alpha_list, progress, "Reschedule..."
                               )
            else:
                alpha_list = list(torch.linspace(0, 1, num_frames))
                print(alpha_list)
                images = morph(alpha_list, progress, "Sampling...")

        return images
