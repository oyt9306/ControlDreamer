import sys

from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from omegaconf import OmegaConf
from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera, get_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from mvdream.annotator.canny import CannyDetector
from mvdream.annotator.midas import MidasDetector
from mvdream.annotator.util import resize_image, HWC3
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image, make_grid


T = torch.Tensor

############################################################################################
#
# StyleAlign Modules
#
############################################################################################

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def expand_first(feat: T, scale=1., ) -> T:
    feat_style = feat
    # feat_style = feat[:, 0].unsqueeze(1).repeat_interleave(feat.shape[1], dim=1)
    # return feat_style
    b = feat_style.shape[0]
    feat_style = torch.stack((feat_style[0], feat_style[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

def register_attention_control(model):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        dim_head = 64
        self.scale = dim_head ** -0.5
        num_frames = 4

        def forward(x, context=None, mask=None):
            q = self.to_q(x)
            if context is not None:
                is_cross = True
            else:
                is_cross = False
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            b, _, _ = q.shape

            if not is_cross:
                q, k, v = map(
                    lambda t: rearrange(t, "b (f l) c -> b f l c", f=num_frames).contiguous(),
                    (q, k, v),
                )
                q = adain(q)
                k = adain(k)
                k = concat_first(k, dim=-2)
                v = concat_first(v, dim=-2)
                q, k, v = map(
                    lambda t: rearrange(t, "b f l c -> b (f l) c", f=num_frames).contiguous(),
                    (q, k, v),
                )

            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )

            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

            if exists(mask):
                raise NotImplementedError
            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            return self.to_out(out)
        return forward


    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    controller = DummyController()


    def register_recr(net_, count, place_in_unet):
        # print(net_.__class__.__name__)
        if 'MemoryEfficientCrossAttention' in net_.__class__.__name__:
            # print(net_.__class__.__name__)
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.model.diffusion_model.named_children()
    for net in sub_nets:
        if "output" in net[0]:
            cross_att_count += register_recr(net[1], 0, "output")
        elif "input" in net[0]:
            cross_att_count += register_recr(net[1], 0, "input")
        elif "middle" in net[0]:
            cross_att_count += register_recr(net[1], 0, "middle")

    controller.num_att_layers = cross_att_count


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

@threestudio.register("multiview-style-control-guidance")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = "sd-v2.1-base-4view" # check mvdream.model_zoo.PRETRAINED_MODELS
        ckpt_path: Optional[str] = None # path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False
        
        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5

        # controlnet
        control_freq : int = 5
        input_mode : str = "depth"
        change_condition: bool = False
        
    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Control Diffusion ...")

        model_ckpt = torch.load(self.cfg.ckpt_path)
        self.model = create_model(self.cfg.model_name).cpu()
        self.model.load_state_dict(model_ckpt['state_dict'], strict=False)

        for p in self.model.parameters():
            p.requires_grad_(False)
        
        threestudio.info(f"Applying Style Align ...")
        register_attention_control(self.model)
        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )
        self.grad_clip_val: Optional[float] = None

        self.to(self.device)
        self.apply_detect = MidasDetector()
        self.apply_canny  = CannyDetector()
        threestudio.info(f"Loaded Multiview StyleAlign Control Diffusion!")

    def static_camera_cond(self, batch):
        '''
        camera sampling for static 4-views in MVdream
        '''
        num_frames = 4
        camera_elev = 15
        camera_azim  = 90
        camera_azim_span = 360

        camera = get_camera(num_frames, 
                            elevation=camera_elev, 
                            azimuth_start=camera_azim, 
                            azimuth_span=camera_azim_span)
        camera = camera.repeat(batch // num_frames,1)
        return camera 
    
    def get_camera_cond(self, 
            camera: Float[Tensor, "B 4 4"],
            fovy = None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation": # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.cfg.camera_condition_type}")
        return camera
    
    def encode_first_stage(self, x):
        return self.model.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return 0.18215 * z
    
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.get_first_stage_encoding(self.encode_first_stage(imgs))
        return latents  # [B, 4, 32, 32] Latent space image

    def normalize_img(self, img):
        img = (img*2 -1)
        return img


    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        idx: int = 1, 
        rgb_as_latents: bool = False,
        fovy = None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        scaling_factor=0.18215,
        **kwargs,
    ): #         
        batch_size = rgb.shape[0]
        camera = c2w
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = F.interpolate(rgb_BCHW, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(rgb_BCHW, (self.cfg.image_size, self.cfg.image_size), mode='bilinear', align_corners=False)
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        detected_maps = []
        for i in range(len(pred_rgb)):
            img_load= to_pil_image(pred_rgb[i].detach().cpu())
            img_detect = np.array(img_load)  
            if self.cfg.input_mode == 'depth':
                detected_map, _ = self.apply_detect(img_detect, 100,200) # MiDas Depth
            elif self.cfg.input_mode == 'canny':
                detected_map = self.apply_canny(img_detect, 100,200) # canny edge
            elif self.cfg.input_mode == 'normal':
                _, detected_map = self.apply_detect(img_detect, bg_th=0.4) # MiDas Normal
            detected_map = torch.from_numpy(HWC3(detected_map).copy()).permute(2,0,1).float().cuda() / 255.0
            detected_maps.append(detected_map)
        input_cond = torch.stack(detected_maps)

        # sample timestep
        if timestep is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2,1).to(text_embeddings)
                context = {"context": text_embeddings, "camera": camera, "num_frames": self.cfg.n_view}
            else:
                context = {"context": text_embeddings}

            # Apply input conditions            
            if not self.cfg.change_condition: 
                context["control"] = torch.cat([input_cond] * 2)
            else:
                if self.global_step < self.cfg.control_freq:
                    context["control"] = torch.cat([input_cond] * 2)
                else:
                    context["control"] = None
                     
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)    
            
        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.cfg.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.cfg.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = self.cfg.recon_std_rescale * latents_recon_adjust + (1-self.cfg.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = (1 - self.alphas_cumprod[t])
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.global_step = global_step
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )