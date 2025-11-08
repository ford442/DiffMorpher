import os
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import safetensors
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import StableDiffusionXLPipeline
from argparse import ArgumentParser
import inspect

from utils.model_utils import get_img, slerp, do_replace_attn 
from utils.lora_utils import train_lora_xl
from utils.alpha_scheduler import AlphaScheduler

class StoreProcessor():
    def __init__(self, original_processor, value_dict, name, num_steps=50, lamd=0.6):
        self.original_processor = original_processor
        self.value_dict = value_dict
        self.name = name
        self.value_dict[self.name] = dict()
        self.id = 0
        # Calculate the step limit based on lamd
        self.limit = int(num_steps * lamd)

    def __call__(self, attn, hidden_states, *args, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            # Only store the map if we are under the step limit
            if self.id < self.limit:
                self.value_dict[self.name][self.id] = hidden_states.detach().cpu()
            # Always increment the ID to match the LoadProcessor
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
                # MODIFIED: Move maps from CPU to the correct GPU device and match dtype just-in-time
                map0 = self.img0_dict[self.name][self.id].to(hidden_states.device, dtype=hidden_states.dtype)
                map1 = self.img1_dict[self.name][self.id].to(hidden_states.device, dtype=hidden_states.dtype)
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
            # Reset the ID when it reaches the end of the 50-step sequence
            if self.id == 50: 
                    self.id = 0
        else:
            res = self.original_processor(attn, hidden_states, *args,
                                          encoder_hidden_states=encoder_hidden_states,
                                          attention_mask=attention_mask,
                                          **kwargs)
        return res


class DiffMorpherPipelineXL(StableDiffusionXLPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder=None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        
        if hasattr(self.text_encoder_2, "config") and self.text_encoder_2.config.projection_dim is not None:
             self.text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        else:
             print("Warning: text_encoder_2.config.projection_dim not found, using fallback 1280.")
             self.text_encoder_projection_dim = 1280 

        self.img0_dict = dict()
        self.img1_dict = dict()
        
    @property
    def components(self):
        comps = {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "unet": self.unet,
            "scheduler": self.scheduler,
        }
        if hasattr(self, "safety_checker") and self.safety_checker is not None:
            comps["safety_checker"] = self.safety_checker
        if hasattr(self, "feature_extractor") and self.feature_extractor is not None:
            comps["feature_extractor"] = self.feature_extractor
        if hasattr(self, "image_encoder") and self.image_encoder is not None:
            comps["image_encoder"] = self.image_encoder
        return comps
    
    @torch.no_grad()
    def image2latent(self, image):
        device = self._device # Use the pipeline's internal device
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        latents = self.vae.encode(image.to(device=device, dtype=self.vae.dtype))['latent_dist'].mean
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        vae_dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)

        # Also cast the latents to float32 to match the VAE
        latents = latents.to(dtype=torch.float32)

        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.vae.dtype))['sample']
        self.vae.to(dtype=vae_dtype)

        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.no_grad()
    def ddim_inversion(self, latent, prompt_embeds, pooled_prompt_embeds, guidance_scale):
        # Force "cuda" as the target device for computation, bypassing self.device issues.
        device = torch.device("cuda")
        unet_dtype = self.unet.dtype # DEFINES the variable

        latent = latent.to(device=device, dtype=unet_dtype)
        prompt_embeds = prompt_embeds.to(device=device, dtype=unet_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=unet_dtype)
        # --- END DTYPE FIX ---

        timesteps = reversed(self.scheduler.timesteps)

        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024),
            dtype=unet_dtype, # Use the correct dtype
            text_encoder_projection_dim=self.text_encoder_projection_dim
        ).to(device)
        
        if guidance_scale > 1.0:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        # --- END NEW ---
        
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
    
        for i, t in enumerate(tqdm.tqdm(timesteps, desc="DDIM inversion")):
            timestep_gpu = t.to(device)

            # --- NEW: Add CFG logic for model_inputs ---
            model_inputs = torch.cat([latent] * 2) if guidance_scale > 1. else latent
            # --- END NEW ---

            eps = self.unet(
                model_inputs, # MODIFIED: Use model_inputs
                timestep_gpu,
                encoder_hidden_states=prompt_embeds, 
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # --- NEW: Perform CFG calculation ---
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = eps.chunk(2, dim=0)
                eps = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # --- END NEW ---
            
            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            
            if prev_timestep >= 0:
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep]
            else:
                alpha_prod_t_prev = self.scheduler.final_alpha_cumprod

            pred_x0 = (latent - (1 - alpha_prod_t) ** 0.5 * eps) / alpha_prod_t ** 0.5
            pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * eps
            latent = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir

        return latent

    @torch.no_grad()
    def cal_latent(self, num_inference_steps, guidance_scale, unconditioning,
                   img_noise_0, img_noise_1,
                   prompt_embeds_0, pooled_embeds_0,
                   prompt_embeds_1, pooled_embeds_1,
                   alpha, use_lora):
        
        device = torch.device("cuda")
        unet_dtype = self.unet.dtype

        latents = slerp(img_noise_0, img_noise_1, alpha, self.use_adain).to(unet_dtype)
    
        prompt_embeds = ((1 - alpha) * prompt_embeds_0 + alpha * prompt_embeds_1).to(unet_dtype)
        pooled_embeds = ((1 - alpha) * pooled_embeds_0 + alpha * pooled_embeds_1).to(unet_dtype)

        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024),
            dtype=unet_dtype, # Use the correct dtype here as well
            text_encoder_projection_dim=self.text_encoder_projection_dim
        ).to(device)
        if guidance_scale > 1.0:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for i, t in enumerate(tqdm.tqdm(self.scheduler.timesteps, desc=f"DDIM Sampler, alpha={alpha:.2f}")):
            model_inputs = torch.cat([latents] * 2) if guidance_scale > 1. else latents
            timestep = t.to(device)
        
            noise_pred = self.unet(
                model_inputs,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents

    @torch.no_grad()
    def get_text_embeddings(self, prompt, guidance_scale, neg_prompt, batch_size):
        device = self._device
        
        if neg_prompt is None:
            neg_prompt = ""

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=[prompt] * batch_size,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=[neg_prompt] * batch_size,
        )

        if guidance_scale > 1.:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            pooled_embeds = pooled_prompt_embeds

        return prompt_embeds, pooled_embeds

    def _morph(self, alpha_list, progress, desc, **kwargs):
        attn_beta = kwargs['attn_beta']
        lamd = kwargs['lamd']
        num_inference_steps = kwargs['num_inference_steps']
        guidance_scale = kwargs['guidance_scale']
        unconditioning = kwargs['unconditioning']
        img_noise_0 = kwargs['img_noise_0']
        img_noise_1 = kwargs['img_noise_1']
        prompt_embeds_0 = kwargs['prompt_embeds_0']
        pooled_embeds_0 = kwargs['pooled_embeds_0']
        prompt_embeds_1 = kwargs['prompt_embeds_1']
        pooled_embeds_1 = kwargs['pooled_embeds_1']
        save_intermediates = kwargs['save_intermediates']
        num_frames = kwargs['num_frames']
        fix_lora = kwargs['fix_lora']
        
        images = []
        
        if attn_beta is not None:
            if self.use_lora:
                self.set_adapters(["lora_0"], adapter_weights=[1.0])

            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if do_replace_attn(k):
                    # MODIFIED: Pass num_inference_steps and lamd to limit RAM usage
                    attn_processor_dict[k] = StoreProcessor(
                        self.unet.attn_processors[k], 
                        self.img0_dict, 
                        k, 
                        num_steps=num_inference_steps, 
                        lamd=lamd
                    )
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]
            self.unet.set_attn_processor(attn_processor_dict)
            
            latents_0 = self.cal_latent(
                num_inference_steps, guidance_scale, unconditioning,
                img_noise_0, img_noise_1,
                prompt_embeds_0, pooled_embeds_0,
                prompt_embeds_1, pooled_embeds_1,
                alpha_list[0], self.use_lora
            )
            first_image = self.latent2image(latents_0)
            first_image = Image.fromarray(first_image)
            if save_intermediates:
                first_image.save(f"{self.output_path}/{0:02d}.png")

            if self.use_lora:
                self.set_adapters(["lora_1"], adapter_weights=[1.0])

            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if do_replace_attn(k):
                    # MODIFIED: Pass num_inference_steps and lamd to limit RAM usage
                    attn_processor_dict[k] = StoreProcessor(
                        self.unet.attn_processors[k], 
                        self.img1_dict, 
                        k, 
                        num_steps=num_inference_steps, 
                        lamd=lamd
                    )
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]
            self.unet.set_attn_processor(attn_processor_dict)
        
            latents_1 = self.cal_latent(
                num_inference_steps, guidance_scale, unconditioning,
                img_noise_0, img_noise_1,
                prompt_embeds_0, pooled_embeds_0,
                prompt_embeds_1, pooled_embeds_1,
                alpha_list[-1], self.use_lora
            )
            last_image = self.latent2image(latents_1)
            last_image = Image.fromarray(last_image)
            if save_intermediates:
                last_image.save(f"{self.output_path}/{num_frames - 1:02d}.png")

            intermediate_images = []
            for i in progress.tqdm(range(1, num_frames - 1), desc=desc):
                alpha = alpha_list[i]
                if self.use_lora:
                    self.set_adapters(["lora_0", "lora_1"], adapter_weights=[1-alpha, alpha])
            
                attn_processor_dict = {}
                for k in self.unet.attn_processors.keys():
                    if do_replace_attn(k):
                        attn_processor_dict[k] = LoadProcessor(
                            self.unet.attn_processors[k], k, self.img0_dict, self.img1_dict, 
                            alpha, attn_beta, lamd
                        )
                    else:
                        attn_processor_dict[k] = self.unet.attn_processors[k]
                self.unet.set_attn_processor(attn_processor_dict)

                latents = self.cal_latent(
                    num_inference_steps, guidance_scale, unconditioning,
                    img_noise_0, img_noise_1,
                    prompt_embeds_0, pooled_embeds_0,
                    prompt_embeds_1, pooled_embeds_1,
                    alpha, self.use_lora
                )
                image = self.latent2image(latents)
                image = Image.fromarray(image)
                if save_intermediates:
                    image.save(f"{self.output_path}/{i:02d}.png")
                intermediate_images.append(image)
            
            images = [first_image] + intermediate_images + [last_image]
        else:
            for i in progress.tqdm(range(num_frames), desc=desc):
                alpha = alpha_list[i]
                if self.use_lora:
                    if fix_lora is not None:
                        adapter_name = "lora_0" if fix_lora == 0 else "lora_1"
                        self.set_adapters([adapter_name], adapter_weights=[1.0])
                    else:
                        self.set_adapters(["lora_0", "lora_1"], adapter_weights=[1 - alpha, alpha])
                
                self.unet.set_default_attn_processor()
                latents = self.cal_latent(
                    num_inference_steps, guidance_scale, unconditioning,
                    img_noise_0, img_noise_1,
                    prompt_embeds_0, pooled_embeds_0,
                    prompt_embeds_1, pooled_embeds_1,
                    alpha, self.use_lora
                )
                image = self.latent2image(latents)
                image = Image.fromarray(image)
                if save_intermediates:
                    image.save(f"{self.output_path}/{i:02d}.png")
                images.append(image)


        return images
        
    def __call__(
        self,
        img_0=None, img_1=None, img_path_0=None, img_path_1=None,
        prompt_0="", prompt_1="", save_lora_dir="./lora_xl",
        load_lora_path_0=None, load_lora_path_1=None,
        lora_steps=200, lora_lr=2e-4, lora_rank=16, batch_size=1,
        height=1024, width=1024, num_inference_steps=50,
        num_actual_inference_steps=None, guidance_scale=7.5,
        attn_beta=0, lamd=0.6, use_lora=True, use_adain=True,
        use_reschedule=True, output_path="./results", num_frames=50,
        fix_lora=None, progress=tqdm, unconditioning=None,
        neg_prompt=None, save_intermediates=False, **kwds
    ):
        self.scheduler.set_timesteps(num_inference_steps)
        self.use_lora = use_lora
        self.use_adain = use_adain
        self.use_reschedule = use_reschedule
        self.output_path = output_path
        
        # This is the most efficient and robust fix for the scheduler's tensors.
        if self._device.type != 'cpu':
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self._device)
            if hasattr(self.scheduler, 'final_alpha_cumprod'):
                 if self.scheduler.final_alpha_cumprod is not None:
                    self.scheduler.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(self._device)

        if img_0 is None: img_0 = Image.open(img_path_0).convert("RGB")
        if img_1 is None: img_1 = Image.open(img_path_1).convert("RGB")
            
        if self.use_lora:
            os.makedirs(save_lora_dir, exist_ok=True)
            lora_path_0 = os.path.join(save_lora_dir, f"{os.path.splitext(os.path.basename(img_path_0))[0]}_lora.safetensors")
            if not os.path.exists(lora_path_0):
                print(f"Training LoRA for image 0...")
                train_lora_xl(
                    image=img_0, prompt=prompt_0, save_path=lora_path_0,
                    unet=self.unet, vae=self.vae, text_encoder=self.text_encoder,
                    text_encoder_2=self.text_encoder_2, tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2, lora_steps=lora_steps,
                    lora_lr=lora_lr, lora_rank=lora_rank
                )

            lora_path_1 = os.path.join(save_lora_dir, f"{os.path.splitext(os.path.basename(img_path_1))[0]}_lora.safetensors")
            if not os.path.exists(lora_path_1):
                print(f"Training LoRA for image 1...")
                train_lora_xl(
                    image=img_1, prompt=prompt_1, save_path=lora_path_1,
                    unet=self.unet, vae=self.vae, text_encoder=self.text_encoder,
                    text_encoder_2=self.text_encoder_2, tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2, lora_steps=lora_steps,
                    lora_lr=lora_lr, lora_rank=lora_rank
                )

            print("Loading and fusing LoRA adapters...")
            self.load_lora_weights(os.path.dirname(lora_path_0), weight_name=os.path.basename(lora_path_0), adapter_name="lora_0")
            self.load_lora_weights(os.path.dirname(lora_path_1), weight_name=os.path.basename(lora_path_1), adapter_name="lora_1")
            self.set_adapters(["lora_0", "lora_1"])
            
        prompt_embeds_0, pooled_embeds_0 = self.get_text_embeddings(prompt_0, guidance_scale, neg_prompt, batch_size)
        prompt_embeds_1, pooled_embeds_1 = self.get_text_embeddings(prompt_1, guidance_scale, neg_prompt, batch_size)
        
        img_0_processed = get_img(img_0)
        img_1_processed = get_img(img_1)
        
        if self.use_lora:
            self.set_adapters(["lora_0"], adapter_weights=[1.0])
        print("Inverting image 0...")
        img_noise_0 = self.ddim_inversion(self.image2latent(img_0_processed), prompt_embeds_0, pooled_embeds_0, guidance_scale)

        if self.use_lora:
            self.set_adapters(["lora_1"], adapter_weights=[1.0])
        print("Inverting image 1...")
        img_noise_1 = self.ddim_inversion(self.image2latent(img_1_processed), prompt_embeds_1, pooled_embeds_1, guidance_scale)

        print("Latent noise vectors calculated.")
        
        morph_kwargs = {
            "attn_beta": attn_beta, "lamd": lamd, "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale, "unconditioning": unconditioning,
            "img_noise_0": img_noise_0, "img_noise_1": img_noise_1,
            "prompt_embeds_0": prompt_embeds_0, "pooled_embeds_0": pooled_embeds_0,
            "prompt_embeds_1": prompt_embeds_1, "pooled_embeds_1": pooled_embeds_1,
            "save_intermediates": save_intermediates, "num_frames": num_frames,
            "fix_lora": fix_lora
        }

        with torch.no_grad():
            if self.use_reschedule:
                alpha_scheduler = AlphaScheduler()
                alpha_list = list(torch.linspace(0, 1, num_frames))
                images_pt = self._morph(alpha_list, progress, "Sampling...", **morph_kwargs)
                images_pt = [transforms.ToTensor()(img).unsqueeze(0) for img in images_pt]
                alpha_scheduler.from_imgs(images_pt)
                alpha_list = alpha_scheduler.get_list()
                print("Rescheduling alphas:", alpha_list)
                images = self._morph(alpha_list, progress, "Resampling with new alphas...", **morph_kwargs)
            else:
                alpha_list = list(torch.linspace(0, 1, num_frames))
                images = self._morph(alpha_list, progress, "Sampling...", **morph_kwargs)
                
        # --- NEW: Final cleanup a_morph_kwargs)
        # Clear the large attention map dictionaries
        self.img0_dict.clear()
        self.img1_dict.clear()
        
        # Force Python's garbage collector to run
        import gc
        gc.collect()
        
        # Clear the PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # --- END NEW ---

        return images
