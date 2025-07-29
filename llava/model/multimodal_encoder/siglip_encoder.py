"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from llava.utils import rank0_print

from transformers import SiglipModel, SiglipProcessor


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("xf-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
class SigLipMultiModalEncoder(nn.Module):
    def __init__(self, model_name, delay_load=False):
        super().__init__()
        self.model_name = model_name
        if not delay_load:
            self.load_model()
    
    def load_model(self, device_map='cuda'):
        """Load the SigLip model and processor"""
        self.model = SiglipModel.from_pretrained(
            self.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.processor = SiglipProcessor.from_pretrained(self.model_name)
        # Store normalization parameters for direct tensor processing
        self.image_mean = torch.tensor(self.processor.image_processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(self.processor.image_processor.image_std).view(1, 3, 1, 1)
        self.target_size = (
            self.processor.image_processor.size["height"], 
            self.processor.image_processor.size["width"]
        )
        
    def prepare_images(self, images, device):
        """Process image tensors directly without PIL conversion"""
        # Ensure images are in the right format [batch, channels, height, width]
        if images.dim() == 3:  # Single image [channels, height, width]
            images = images.unsqueeze(0)
            
        # Resize if needed
        current_size = images.shape[2:4]
        if current_size != self.target_size:
            images = torch.nn.functional.interpolate(
                images, 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply normalization directly
        self.image_mean = self.image_mean.to(device)
        self.image_std = self.image_std.to(device)
        
        # Check if normalization is needed (normalize from [0,1] or [-1,1] to model's expected range)
        if images.min() < 0 or images.max() > 1:
            # Assuming images are in range [-1, 1], convert to [0, 1]
            images = (images + 1) / 2
            
        # Apply model's normalization
        images = (images - self.image_mean) / self.image_std
        
        return images
    
    def prepare_text(self, text, device):
        """Process text input"""
        if isinstance(text, str):
            text = [text]  # Convert single string to list for batch processing
            
        text_inputs = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.processor.tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        return text_inputs
    
    def encode_input(self, images, text, device_map='cuda'):
        """
        Calculate cosine similarity between image and text embeddings
        
        Args:
            images: Tensor of shape [batch, 3, height, width]
            text: String or list of strings
            device_map: Device to run the model on
        
        Returns:
            Tensor of cosine similarity scores
        """
        device = torch.device(device_map)
        
        # Process images directly as tensors
        processed_images = self.prepare_images(images, device)
        
        # Process text
        text_inputs = self.prepare_text(text, device)
        
        # Create model inputs
        inputs = {
            "pixel_values": processed_images,
            "input_ids": text_inputs.input_ids,
        }

        # Add attention_mask if it exists
        if "attention_mask" in text_inputs:
            inputs["attention_mask"] = text_inputs["attention_mask"]
        
        # Run model inference
        with torch.no_grad():
            with torch.autocast(device_map):
                outputs = self.model(**inputs)
                
        # Extract image and text embeddings from the model outputs
        image_embeds = outputs.vision_model_output.pooler_output
        text_embeds = outputs.text_model_output.pooler_output
        
        # Calculate cosine similarity
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        
        return image_embeds, text_embeds


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size


import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPMultiModalEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", delay_load=False):
        super().__init__()
        self.model_name = model_name
        if not delay_load:
            self.load_model()
    
    def load_model(self, device_map='cuda'):
        """Load the CLIP model and processor"""
        self.model = CLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # Store normalization parameters for direct tensor processing
        # CLIP uses different normalization values than SigLip
        self.image_mean = torch.tensor(self.processor.image_processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(self.processor.image_processor.image_std).view(1, 3, 1, 1)
        
        # CLIP uses 'size' field differently - it's a single integer for square images
        if isinstance(self.processor.image_processor.size, dict):
            self.target_size = (
                self.processor.image_processor.size["shortest_edge"], 
                self.processor.image_processor.size["shortest_edge"]
            )
        else:
            # CLIP typically uses square images
            size = self.processor.image_processor.size
            self.target_size = (size, size)
        
    def prepare_images(self, images, device):
        """Process image tensors directly without PIL conversion"""
        # Ensure images are in the right format [batch, channels, height, width]
        if images.dim() == 3:  # Single image [channels, height, width]
            images = images.unsqueeze(0)
            
        # Resize if needed
        current_size = images.shape[2:4]
        if current_size != self.target_size:
            images = torch.nn.functional.interpolate(
                images, 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply normalization directly
        self.image_mean = self.image_mean.to(device)
        self.image_std = self.image_std.to(device)
        
        # Check if normalization is needed (normalize from [0,1] or [-1,1] to model's expected range)
        if images.min() < 0 or images.max() > 1:
            # Assuming images are in range [-1, 1], convert to [0, 1]
            images = (images + 1) / 2
            
        # Apply model's normalization
        images = (images - self.image_mean) / self.image_std
        
        return images
    
    def prepare_text(self, text, device):
        """Process text input"""
        if isinstance(text, str):
            text = [text]  # Convert single string to list for batch processing
            
        text_inputs = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.processor.tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        return text_inputs
    
    def encode_input(self, images, text, device_map='cuda'):
        """
        Calculate cosine similarity between image and text embeddings
        
        Args:
            images: Tensor of shape [batch, 3, height, width]
            text: String or list of strings
            device_map: Device to run the model on
        
        Returns:
            Tuple of (image_embeds, text_embeds) - normalized embeddings
        """
        device = torch.device(device_map)
        
        # Process images directly as tensors
        processed_images = self.prepare_images(images, device)
        
        # Process text
        text_inputs = self.prepare_text(text, device)
        
        # Create model inputs
        inputs = {
            "pixel_values": processed_images,
            "input_ids": text_inputs.input_ids,
        }

        # Add attention_mask if it exists
        if "attention_mask" in text_inputs:
            inputs["attention_mask"] = text_inputs["attention_mask"]
        
        # Run model inference
        with torch.no_grad():
            with torch.autocast(device_map):
                outputs = self.model(**inputs)
                
        # Extract image and text embeddings from the model outputs
        # For CLIP, the embeddings are directly available in the outputs
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # CLIP outputs are already normalized, but let's ensure they are
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        
        return image_embeds, text_embeds
    
    def get_similarity(self, images, text, device_map='cuda'):
        """
        Calculate cosine similarity between image and text embeddings
        
        Args:
            images: Tensor of shape [batch, 3, height, width]
            text: String or list of strings
            device_map: Device to run the model on
        
        Returns:
            Tensor of cosine similarity scores
        """
        image_embeds, text_embeds = self.encode_input(images, text, device_map)
        
        # Calculate cosine similarity (since embeddings are normalized, this is just dot product)
        similarity = torch.matmul(image_embeds, text_embeds.T)
        
        return similarity


import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F

class DINOVisionEncoder(nn.Module):
    def __init__(self, model_name="facebook/dino-vitb16", delay_load=False):
        """
        DINO Vision Encoder for extracting visual features
        
        Args:
            model_name: DINO model variant to use
                - "facebook/dino-vitb16" (ViT-B/16)
                - "facebook/dino-vits16" (ViT-S/16) 
                - "facebook/dino-vitb8" (ViT-B/8)
                - "facebook/dino-vits8" (ViT-S/8)
            delay_load: Whether to delay model loading
        """
        super().__init__()
        self.model_name = model_name
        if not delay_load:
            self.load_model()
    
    def load_model(self, device_map='cuda'):
        """Load the DINO model and processor"""
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        # Store normalization parameters for direct tensor processing
        self.image_mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        
        # Get target image size
        if hasattr(self.processor, 'size'):
            if isinstance(self.processor.size, dict):
                # Handle different size format variations
                if 'shortest_edge' in self.processor.size:
                    size = self.processor.size['shortest_edge']
                elif 'height' in self.processor.size:
                    size = self.processor.size['height']  # Assuming square
                else:
                    size = 224  # Default DINO size
            else:
                size = self.processor.size
        else:
            size = 224  # Default fallback
            
        self.target_size = (size, size)
        
        # Get the embedding dimension from the model config
        self.embed_dim = self.model.config.hidden_size
        
    def prepare_images(self, images, device):
        """Process image tensors directly without PIL conversion"""
        # Ensure images are in the right format [batch, channels, height, width]
        if images.dim() == 3:  # Single image [channels, height, width]
            images = images.unsqueeze(0)
            
        # Resize if needed
        current_size = images.shape[2:4]
        if current_size != self.target_size:
            images = F.interpolate(
                images, 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Move normalization tensors to the correct device
        self.image_mean = self.image_mean.to(device)
        self.image_std = self.image_std.to(device)
        
        # Normalize pixel values to [0, 1] if they're not already
        if images.min() < 0 or images.max() > 1:
            # Assuming images are in range [-1, 1], convert to [0, 1]
            images = (images + 1) / 2
            
        # Apply model's normalization (ImageNet stats)
        images = (images - self.image_mean) / self.image_std
        
        return images
    
    def encode_images(self, images, device_map='cuda', return_patch_tokens=False):
        """
        Extract DINO features from images
        
        Args:
            images: Tensor of shape [batch, 3, height, width]
            device_map: Device to run the model on
            return_patch_tokens: Whether to return patch tokens in addition to CLS token
        
        Returns:
            If return_patch_tokens=False: Tensor of shape [batch, embed_dim] (CLS token features)
            If return_patch_tokens=True: Tuple of (cls_features, patch_features)
                - cls_features: [batch, embed_dim]
                - patch_features: [batch, num_patches, embed_dim]
        """
        device = torch.device(device_map)
        
        # Process images
        processed_images = self.prepare_images(images, device)
        
        # Run model inference
        with torch.no_grad():
            with torch.autocast(device_type=device.type):
                outputs = self.model(pixel_values=processed_images)
        
        # Extract features
        # DINO outputs: last_hidden_state contains [CLS] token + patch tokens
        # CLS token is at index 0
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, embed_dim]
        
        # Extract CLS token features (global image representation)
        cls_features = last_hidden_state[:, 0, :]  # [batch, embed_dim]
        
        if return_patch_tokens:
            # Extract patch token features (spatial features)
            patch_features = last_hidden_state[:, 1:, :]  # [batch, num_patches, embed_dim]
            return cls_features, patch_features
        else:
            return cls_features
    
    def get_similarity(self, images1, images2, device_map='cuda', similarity_type='cosine'):
        """
        Calculate similarity between two sets of images using DINO features
        
        Args:
            images1: First set of images [batch1, 3, height, width]
            images2: Second set of images [batch2, 3, height, width]
            device_map: Device to run the model on
            similarity_type: 'cosine' or 'euclidean'
        
        Returns:
            Similarity matrix [batch1, batch2]
        """
        # Extract features for both image sets
        features1 = self.encode_images(images1, device_map)
        features2 = self.encode_images(images2, device_map)
        
        if similarity_type == 'cosine':
            # Normalize features for cosine similarity
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            
            # Cosine similarity via matrix multiplication
            similarity = torch.matmul(features1, features2.T)
            
        elif similarity_type == 'euclidean':
            # Compute pairwise euclidean distances
            # Expand dimensions for broadcasting
            features1_expanded = features1.unsqueeze(1)  # [batch1, 1, embed_dim]
            features2_expanded = features2.unsqueeze(0)  # [1, batch2, embed_dim]
            
            # Calculate euclidean distance
            distances = torch.norm(features1_expanded - features2_expanded, dim=2)
            
            # Convert distances to similarities (higher = more similar)
            similarity = 1 / (1 + distances)
            
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        return similarity
    
    def extract_patch_features(self, images, device_map='cuda', layer_idx=-1):
        """
        Extract spatial patch features from a specific layer
        Useful for dense prediction tasks or attention visualization
        
        Args:
            images: Tensor of shape [batch, 3, height, width]
            device_map: Device to run the model on
            layer_idx: Which transformer layer to extract features from (-1 for last layer)
        
        Returns:
            Patch features [batch, num_patches, embed_dim]
        """
        device = torch.device(device_map)
        processed_images = self.prepare_images(images, device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type):
                if layer_idx == -1:
                    # Use the standard forward pass for last layer
                    outputs = self.model(pixel_values=processed_images)
                    patch_features = outputs.last_hidden_state[:, 1:, :]
                else:
                    # Get intermediate layer outputs
                    outputs = self.model(
                        pixel_values=processed_images, 
                        output_hidden_states=True
                    )
                    # hidden_states includes embedding layer + all transformer layers
                    patch_features = outputs.hidden_states[layer_idx + 1][:, 1:, :]
        
        return patch_features
    
    def get_attention_maps(self, images, device_map='cuda', head_fusion='mean'):
        """
        Extract attention maps from DINO model
        Useful for visualizing what the model is focusing on
        
        Args:
            images: Tensor of shape [batch, 3, height, width]
            device_map: Device to run the model on
            head_fusion: How to combine attention heads ('mean', 'max', 'min')
        
        Returns:
            Attention maps [batch, num_patches] (attention from CLS token to patches)
        """
        device = torch.device(device_map)
        processed_images = self.prepare_images(images, device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type):
                outputs = self.model(
                    pixel_values=processed_images, 
                    output_attentions=True
                )
        
        # Get attention weights from the last layer
        # Shape: [batch, num_heads, seq_len, seq_len]
        attention_weights = outputs.attentions[-1]
        
        # Extract attention from CLS token (index 0) to patch tokens
        cls_attention = attention_weights[:, :, 0, 1:]  # [batch, num_heads, num_patches]
        
        # Fuse attention heads
        if head_fusion == 'mean':
            attention_maps = cls_attention.mean(dim=1)
        elif head_fusion == 'max':
            attention_maps = cls_attention.max(dim=1)[0]
        elif head_fusion == 'min':
            attention_maps = cls_attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion method: {head_fusion}")
        
        return attention_maps