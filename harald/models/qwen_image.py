"""Qwen-Image pipeline wrapper with slot-injection support."""

from typing import Optional, Tuple
import sys

import torch
from PIL import Image
from diffusers import DiffusionPipeline


class QwenImagePipeline:
    """
    Wrapper for Qwen-Image Diffusion Pipeline with slot-based injection support.

    Provides:
    - Text encoding to prompt embeddings
    - Image generation from prompt embeddings (not strings)
    - Access to pipeline components (tokenizer, text_encoder, unet, vae, scheduler)
    - Device/dtype consistency validation
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Image",
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Qwen-Image pipeline.

        Args:
            model_name: HuggingFace model ID
            device: Target device (must be CUDA)
            dtype: Target dtype (bf16/fp16/fp32)
            hf_token: HuggingFace token for private models
            cache_dir: Cache directory for model files

        Raises:
            SystemExit: If device is not CUDA or pipeline loading fails
        """
        if device.type != "cuda":
            print(
                f"ERROR: Qwen-Image requires CUDA device, got {device}",
                file=sys.stderr,
            )
            sys.exit(1)

        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        print(f"Loading Qwen-Image pipeline: {model_name}")
        print(f"  Device: {device}")
        print(f"  dtype:  {dtype}")

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                token=hf_token,
                cache_dir=cache_dir,
            ).to(device)

            self.pipe.set_progress_bar_config(disable=True)
            print("Qwen-Image pipeline loaded successfully")

        except Exception as e:
            print(
                f"ERROR: Failed to load Qwen-Image pipeline.\n"
                f"  Model: {model_name}\n"
                f"  Error: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate required components
        self._validate_components()

    def _validate_components(self):
        """Validate that pipeline has required components."""
        # Qwen-Image uses transformer (MMDiT) instead of unet
        required = ["tokenizer", "text_encoder", "transformer", "vae", "scheduler"]
        missing = []

        for component in required:
            if not hasattr(self.pipe, component):
                missing.append(component)

        if missing:
            print(
                f"ERROR: Qwen-Image pipeline missing required components: {missing}",
                file=sys.stderr,
            )
            sys.exit(1)

    @property
    def tokenizer(self):
        """Access tokenizer component."""
        return self.pipe.tokenizer

    @property
    def text_encoder(self):
        """Access text encoder component."""
        return self.pipe.text_encoder

    @property
    def transformer(self):
        """Access Transformer (MMDiT) component."""
        return self.pipe.transformer

    @property
    def vae(self):
        """Access VAE component."""
        return self.pipe.vae

    @property
    def scheduler(self):
        """Access scheduler component."""
        return self.pipe.scheduler

    def get_max_seq_length(self) -> int:
        """
        Get maximum sequence length from tokenizer.

        Returns:
            Maximum sequence length
        """
        return self.tokenizer.model_max_length

    @torch.inference_mode()
    def encode_text(
        self,
        prompt: str,
        max_sequence_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt to encode
            max_sequence_length: Maximum sequence length (default: 512)
            padding: Padding mode (default: "max_length")
            truncation: Whether to truncate (default: True)

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask):
                - prompt_embeds: [1, S, H] where S=seq_len, H=hidden_dim
                - prompt_embeds_mask: [1, S]

        Notes:
            - Uses pipeline's encode_prompt() method
            - Embeddings are on self.device with pipeline's native dtype
            - Mask is converted to self.device
        """
        # Use official encode_prompt method from pipeline
        prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        # Ensure mask is on correct device (may return as CPU)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        # Validate shapes
        if prompt_embeds.ndim != 3:
            print(
                f"ERROR: encode_text() returned unexpected shape.\n"
                f"  Expected: [1, S, H] (3D)\n"
                f"  Got:      {prompt_embeds.shape}",
                file=sys.stderr,
            )
            sys.exit(1)

        return prompt_embeds, prompt_embeds_mask

    @torch.inference_mode()
    def generate(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate image from prompt embeddings.

        Args:
            prompt_embeds: Prompt embeddings [1, S, H]
            prompt_embeds_mask: Prompt embeddings mask [1, S]
            negative_prompt_embeds: Optional negative prompt embeddings [1, S, H]
            negative_prompt_embeds_mask: Optional negative prompt embeddings mask [1, S]
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: CFG scale (Qwen-Image uses true_cfg_scale)
            height: Output image height (default: 1024)
            width: Output image width (default: 1024)
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image

        Notes:
            - Does NOT accept string prompts (only embeddings)
            - Uses Qwen-Image specific parameter: true_cfg_scale
            - Generator device must match pipeline device
        """
        # Validate prompt_embeds device/dtype (normalize: cuda == cuda:0)
        if prompt_embeds.device.type != self.device.type:
            print(
                f"ERROR: prompt_embeds device mismatch.\n"
                f"  Expected: {self.device} (type: {self.device.type})\n"
                f"  Got:      {prompt_embeds.device} (type: {prompt_embeds.device.type})",
                file=sys.stderr,
            )
            sys.exit(1)

        # Create generator on correct device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate image using prompt embeddings
        try:
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                true_cfg_scale=guidance_scale,  # Qwen-Image specific
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
            )

            img = result.images[0]
            return img

        except Exception as e:
            print(
                f"ERROR: Image generation failed.\n"
                f"  prompt_embeds shape: {prompt_embeds.shape}\n"
                f"  steps: {num_inference_steps}\n"
                f"  guidance: {guidance_scale}\n"
                f"  size: {height}x{width}\n"
                f"  Error: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    def encode_negative_prompt(
        self,
        negative_prompt: str = " ",
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode negative prompt to embeddings.

        Args:
            negative_prompt: Negative prompt text (default: " " - empty string)
            max_sequence_length: Maximum sequence length

        Returns:
            Tuple of (negative_prompt_embeds, negative_prompt_embeds_mask)
        """
        return self.encode_text(negative_prompt, max_sequence_length)
