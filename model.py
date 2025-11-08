import os
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers, DDPMScheduler
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

from utils.model_utils import get_img, slerp, do_replace_attn 
from utils.lora_utils import train_lora_xl
from utils.alpha_scheduler import AlphaScheduler

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


class DiffMorpherPipelineXL(StableDiffusionXLPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
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
        
        # --- FIX: Manually set the projection dim ---
        # This value is required by SDXL but isn't set
        # by the super().__init__ when passed components manually.
        if hasattr(self.text_encoder_2, "config") and self.text_encoder_2.config.projection_dim is not None:
             self.text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        else:
             # Fallback value if config is somehow missing, 1280 is standard for SDXL
             print("Warning: text_encoder_2.config.projection_dim not found, using fallback 1280.")
             self.text_encoder_projection_dim = 1280 
        # --- END FIX ---

        self.img0_dict = dict()
        self.img1_dict = dict()
        
    @property
    def components(self):
        """
        Forcefully return ONLY the components that are modules
        to bypass the config introspection bug.
        """
        comps = {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "unet": self.unet,
            "scheduler": self.scheduler,
        }

        # Add optional components only if they exist
        if hasattr(self, "safety_checker") and self.safety_checker is not None:
            comps["safety_checker"] = self.safety_checker
        if hasattr(self, "feature_extractor") and self.feature_extractor is not None:
            comps["feature_extractor"] = self.feature_extractor
        if hasattr(self, "image_encoder") and self.image_encoder is not None:
            comps["image_encoder"] = self.image_encoder

        return comps
    
    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps //
                       self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0)
        # input image density range [-1, 1]
        latents = self.vae.encode(image.to(device=DEVICE, dtype=self.vae.dtype))['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]
        
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        """
        predict the sample of the next step in the denoise process.
        """
        prev_timestep = timestep - \
            self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0   
        
    # SDXL Change: ddim_inversion needs to be updated for dual encoders
    @torch.no_grad()
    def ddim_inversion(self, latent, prompt_embeds, pooled_prompt_embeds):
        # --- START ORIGINAL FIX ---
        # Ensure all input tensors to the UNet are on the correct device.
        device = self.device
        latent = latent.to(device)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        # --- END ORIGINAL FIX ---

        timesteps = reversed(self.scheduler.timesteps)
    
        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype, text_encoder_projection_dim=self.text_encoder_projection_dim
        ).to(device)
    
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
    
        # The original code had a bug here, this is the corrected loop
        for i, t in enumerate(tqdm.tqdm(timesteps, desc="DDIM inversion")):
            # 1. predict noise
            eps = self.unet(
                latent, 
                t, 
                encoder_hidden_states=prompt_embeds, 
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # --- START NEW FIX ---
            # Move eps (UNet output) to the correct device *before* using it.
            # With offloading, eps is often returned on the CPU.
            eps = eps.to(device)
            # --- END NEW FIX ---

            # 2. get previous timestep
            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        
            # 3. compute alphas, ensuring they are *also* on the correct device
            # --- START NEW FIX ---
            # self.scheduler.alphas_cumprod is a CPU tensor, so move the value to the device
            alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device)
            
            # Ensure *both* possible values for alpha_prod_t_prev are moved to the device
            if prev_timestep >= 0:
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep].to(device)
            else:
                # self.scheduler.final_alpha_cumprod is also a tensor that lives on the CPU
                alpha_prod_t_prev = self.scheduler.final_alpha_cumprod.to(device)
            # --- END NEW FIX ---
        
            # 4. compute predicted original sample from predicted noise
            # This line will now work, as latent, eps, and alpha_prod_t are all on 'device'
            pred_x0 = (latent - (1 - alpha_prod_t) ** 0.5 * eps) / alpha_prod_t ** 0.5
        
            # 5. compute direction pointing to x_t
            # This also now works, as alpha_prod_t_prev and eps are on the same device
            pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * eps
        
            # 6. compute x_t-1
            latent = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir

        return latent

    # SDXL Change: cal_latent needs to interpolate both sets of embeddings
    @torch.no_grad()
    def cal_latent(self, num_inference_steps, guidance_scale, unconditioning,
                   img_noise_0, img_noise_1,
                   prompt_embeds_0, pooled_embeds_0,
                   prompt_embeds_1, pooled_embeds_1,
                   alpha, use_lora): # No more lora_0, lora_1, fix_lora
        
        latents = slerp(img_noise_0, img_noise_1, alpha, self.use_adain)
        
        prompt_embeds = (1 - alpha) * prompt_embeds_0 + alpha * prompt_embeds_1
        pooled_embeds = (1 - alpha) * pooled_embeds_0 + alpha * pooled_embeds_1

        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype, text_encoder_projection_dim=self.text_encoder_projection_dim
        ).to(self.device)
        
        # Correct handling of CFG for pooled embeddings
        if guidance_scale > 1.:
            # The incoming prompt_embeds and pooled_embeds are already concatenated for CFG
            uncond_pooled, cond_pooled = pooled_embeds.chunk(2)
            # Interpolate for the final conditional and unconditional pooled embeddings
            pooled_embeds = torch.cat([uncond_pooled, cond_pooled])

        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        # LoRA is now handled outside this function using set_adapters
        
        for i, t in enumerate(tqdm.tqdm(self.scheduler.timesteps, desc=f"DDIM Sampler, alpha={alpha:.2f}")):
            model_inputs = torch.cat([latents] * 2) if guidance_scale > 1. else latents
            timestep = t.to(self.device)
            noise_pred = self.unet(
                model_inputs,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)
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

    def __call__(
            self,
            img_0=None,
            img_1=None,
            img_path_0=None,
            img_path_1=None,
            prompt_0="",
            prompt_1="",
            save_lora_dir="./lora",
            load_lora_path_0=None, # This will be ignored, but we leave it for API compatibility
            load_lora_path_1=None, # This will be ignored, but we leave it for API compatibility
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
                
        if self.device.type != 'cpu':
             self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        if img_0 is None: img_0 = Image.open(img_path_0).convert("RGB")
        if img_1 is None: img_1 = Image.open(img_path_1).convert("RGB")
            
        final_lora_path_0 = load_lora_path_0
        final_lora_path_1 = load_lora_path_1

        if self.use_lora:
            os.makedirs(save_lora_dir, exist_ok=True)
            
            # --- Handle LoRA for Image 0 ---
            # Create a directory name, not a file name
            lora_dir_name_0 = f"{os.path.splitext(os.path.basename(img_path_0))[0]}_lora.safetensors"
            final_lora_path_0 = os.path.join(save_lora_dir, lora_dir_name_0)

            if not os.path.exists(final_lora_path_0):
                print(f"Training LoRA for image 0...")
                lora_state_dict = train_lora_xl(
                    image=img_0, prompt=prompt_0, save_path=final_lora_path_0,
                    unet=self.unet, vae=self.vae,
                    text_encoder=self.text_encoder, text_encoder_2=self.text_encoder_2,
                    tokenizer=self.tokenizer, tokenizer_2=self.tokenizer_2,
                    lora_steps=lora_steps, lora_lr=lora_lr, lora_rank=lora_rank
                )
                # --- The 'lora_state_dict' variable is no longer needed. ---
                # --- REMOVE the self.save_lora_weights block. ---
                # self.save_lora_weights(
                #     save_directory=os.path.dirname(final_lora_path_0),
                #     weight_name=os.path.basename(final_lora_path_0),
                #     unet_lora_layers=lora_state_dict
                # )
                # print(f"LoRA for image 0 saved to {final_lora_path_0}")

            # --- Handle LoRA for Image 1 ---
            lora_dir_name_1 = f"{os.path.splitext(os.path.basename(img_path_1))[0]}_lora.safetensors"
            final_lora_path_1 = os.path.join(save_lora_dir, lora_dir_name_1)

            if not os.path.exists(final_lora_path_1):
                print(f"Training LoRA for image 1...")
                lora_state_dict = train_lora_xl(
                    image=img_1, prompt=prompt_1, save_path=final_lora_path_1,
                    unet=self.unet, vae=self.vae,
                    text_encoder=self.text_encoder, text_encoder_2=self.text_encoder_2,
                    tokenizer=self.tokenizer, tokenizer_2=self.tokenizer_2,
                    lora_steps=lora_steps, lora_lr=lora_lr, lora_rank=lora_rank
                )
                # --- The 'lora_state_dict' variable is no longer needed. ---
                # --- REMOVE the self.save_lora_weights block. ---
                # self.save_lora_weights(
                #     save_directory=os.path.dirname(final_lora_path_1),
                #     weight_name=os.path.basename(final_lora_path_1),
                #     unet_lora_layers=lora_state_dict
                # )
                # print(f"LoRA for image 1 saved to {final_lora_path_1}")


            # --- THE FIX: Load from the directories ---
            print("Loading and fusing LoRA adapters...")
            self.load_lora_weights(
                os.path.dirname(final_lora_path_0),
                weight_name=os.path.basename(final_lora_path_0),
                adapter_name="lora_0"
            )
            self.load_lora_weights(
                os.path.dirname(final_lora_path_1),
                weight_name=os.path.basename(final_lora_path_1),
                adapter_name="lora_1"
            )
    
            # Now, the pipeline's 'set_adapters' method will find the adapters
            # that we have successfully loaded directly onto its UNet component.
            self.set_adapters(["lora_0", "lora_1"])
            
        # SDXL Change: Get both sets of embeddings
        prompt_embeds_0, pooled_embeds_0 = self.get_text_embeddings(
            prompt_0, guidance_scale, neg_prompt, batch_size)
        prompt_embeds_1, pooled_embeds_1 = self.get_text_embeddings(
            prompt_1, guidance_scale, neg_prompt, batch_size)
        
        img_0 = get_img(img_0) # Uses get_img from model_utils_xl (1024)
        img_1 = get_img(img_1) # Uses get_img from model_utils_xl (1024)
        
        if self.use_lora:
            # Ensure the correct LoRA is active for the inversion
            self.set_adapters(["lora_0"], adapter_weights=[1.0])

        # Slice the tensors to get only the conditional embeddings
        cond_prompt_embeds_0 = prompt_embeds_0[1:2]
        cond_pooled_embeds_0 = pooled_embeds_0[1:2]

        print("Inverting image 0...")
        img_noise_0 = self.ddim_inversion(
            self.image2latent(img_0), cond_prompt_embeds_0, cond_pooled_embeds_0)


        # 2. Perform DDIM Inversion for the second image (where the original error occurred)
        if self.use_lora:
            # Switch to the LoRA for the second image
            self.set_adapters(["lora_1"], adapter_weights=[1.0])

        # Slice the tensors to get only the conditional embeddings
        cond_prompt_embeds_1 = prompt_embeds_1[1:2]
        cond_pooled_embeds_1 = pooled_embeds_1[1:2]

        print("Inverting image 1...")
        img_noise_1 = self.ddim_inversion(
            self.image2latent(img_1), cond_prompt_embeds_1, cond_pooled_embeds_1)

        # --- END FIX ---

        # The code from this point was trying to use img_noise_0 and img_noise_1
        # and should now work correctly.
        print("latents shape: ", img_noise_0.shape)
        
        original_processor = list(self.unet.attn_processors.values())[0]
        
    def morph(alpha_list, progress, desc):
        """
        Generates the morphing sequence.
        This function now uses the modern `set_adapters` API for LoRA interpolation
        while retaining the optional attention-sharing mechanism.
        """
        images = []
    
        # This branch enables the attention-sharing mechanism for smoother transitions.
        if attn_beta is not None and attn_beta > 0:
        
            # --- 1. Generate the first frame (alpha=0) and store its attention maps ---
            if self.use_lora:
                # Set LoRA to be 100% the first image's adapter
                self.set_adapters(["lora_0"], adapter_weights=[1.0])

            # Prepare the UNet to store attention maps from the first image
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if do_replace_attn(k):
                    attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k], self.img0_dict, k)
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]
            self.unet.set_attn_processor(attn_processor_dict)

            # Calculate the latent for the first image
            latents_0 = self.cal_latent(
                num_inference_steps, guidance_scale, unconditioning,
                img_noise_0, img_noise_1,
                prompt_embeds_0, pooled_embeds_0,
                prompt_embeds_1, pooled_embeds_1,
                alpha_list[0], self.use_lora # alpha = 0
            )
            first_image = self.latent2image(latents_0)
            first_image = Image.fromarray(first_image)
            if save_intermediates:
                first_image.save(f"{self.output_path}/{0:02d}.png")

            # --- 2. Generate the last frame (alpha=1) and store its attention maps ---
            if self.use_lora:
                # Set LoRA to be 100% the second image's adapter
                self.set_adapters(["lora_1"], adapter_weights=[1.0])

            # Prepare the UNet to store attention maps from the second image
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if do_replace_attn(k):
                    attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k], self.img1_dict, k)
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]
            self.unet.set_attn_processor(attn_processor_dict)
        
            # Calculate the latent for the last image
            latents_1 = self.cal_latent(
                num_inference_steps, guidance_scale, unconditioning,
                img_noise_0, img_noise_1,
                prompt_embeds_0, pooled_embeds_0,
                prompt_embeds_1, pooled_embeds_1,
                alpha_list[-1], self.use_lora # alpha = 1
            )
            last_image = self.latent2image(latents_1)
            last_image = Image.fromarray(last_image)
            if save_intermediates:
                last_image.save(f"{self.output_path}/{num_frames - 1:02d}.png")

            # --- 3. Generate intermediate frames using LoRA interpolation and attention sharing ---
            intermediate_images = []
            for i in progress.tqdm(range(1, num_frames - 1), desc=desc):
                alpha = alpha_list[i]
            
                if self.use_lora:
                    # The modern way to interpolate: set weights on the two named adapters
                    self.set_adapters(["lora_0", "lora_1"], adapter_weights=[1-alpha, alpha])
            
                # Prepare the UNet to load and blend the stored attention maps
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

                # Calculate the latent for the intermediate image
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
            # This is the simpler path: generate all frames without attention sharing.
            # Morphing comes from noise, prompt, and LoRA interpolation only.
            for i in progress.tqdm(range(num_frames), desc=desc):
                alpha = alpha_list[i]
            
                if self.use_lora:
                    # Interpolate the LoRAs for the current frame
                    if fix_lora is not None:
                        # If fixing LoRA, only use one adapter at full strength
                        adapter_name = "lora_0" if fix_lora == 0 else "lora_1"
                        self.set_adapters([adapter_name], adapter_weights=[1.0])
                    else:
                        # Otherwise, blend the two adapters based on alpha
                        self.set_adapters(["lora_0", "lora_1"], adapter_weights=[1 - alpha, alpha])
            
                # Ensure the original attention processors are active
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
        
        # (Reschedule logic is identical and correct)
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
