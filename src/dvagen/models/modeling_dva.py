import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    Cache,
    GenerationMixin,
    LogitsProcessor,
    PreTrainedModel,
)
from transformers.activations import get_activation
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import LossKwargs
from typing_extensions import Unpack

from .configuration_dva import DVAConfig
from ..utils.info_nce_loss import InfoNCE


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# TODO DVAEmbeddingModel: sv embeddings (remove copy?) + phrase encoder
# class DVAEmbeddingModel(PreTrainedModel):


# extract features in phrase encoder
class DVAModel(PreTrainedModel, GenerationMixin):
    config_class = DVAConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: DVAConfig):
        super().__init__(config)
        self.config = config
        if (
            self.config.text_encoder_config is not None and
            self.config.language_model_config is not None and
            self.config.phrase_encoder_config is not None
        ):
            # Initialize the DVAModel from scratch
            self.text_encoder = AutoModel.from_config(config.text_encoder_config)
            self.language_model = AutoModelForCausalLM.from_config(config.language_model_config)
            self.phrase_encoder = AutoModel.from_config(config.phrase_encoder_config)
            self.sv_input_embeddings = self._build_sv_embeddings(self.language_model.get_input_embeddings())
            self.sv_output_embeddings = self._build_sv_embeddings(self.language_model.get_output_embeddings())
            if self.config.use_text_encoder_proj:
                self.text_encoder_proj = self._build_text_encoder_proj()
            if self.config.use_phrase_encoder_proj:
                self.phrase_encoder_proj = self._build_phrase_encoder_proj()
            if self.config.use_type_loss:
                self.type_classification_head = self._build_type_classification_head()
            if self.config.use_description_loss:
                self.description_proj = self._build_description_proj()
        self.loss_type = "ForCausalLM"
        if self.config.use_type_loss:
            self.type_loss = self.set_type_loss()
            self.type_loss_weight = self.config.type_loss_weight
        if self.config.use_description_loss:
            self.description_loss = InfoNCE()
            self.description_loss_weight = self.config.description_loss_weight
        self.vocab_size = self.config.language_model_config.vocab_size

        self.post_init()

    @staticmethod
    def _build_sv_embeddings(src_embeddings: nn.Module) -> nn.Parameter:
        return nn.Parameter(src_embeddings.weight.clone().detach())

    def _build_text_encoder_proj(self) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(self.config.text_encoder_proj_pdrop),
            get_activation(self.config.text_encoder_proj_act),
            nn.Linear(self.text_encoder.config.hidden_size, self.language_model.config.hidden_size),
        )

    def _build_phrase_encoder_proj(self) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(self.config.phrase_encoder_proj_pdrop),
            get_activation(self.config.phrase_encoder_proj_act),
            nn.Linear(self.phrase_encoder.config.hidden_size, self.language_model.config.hidden_size),
        )

    def _build_type_classification_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(self.config.type_classification_head_pdrop),
            get_activation(self.config.type_classification_head_act),
            nn.Linear(self.phrase_encoder.config.hidden_size, self.config.type_classification_head_num_classes),
        )

    def _build_description_proj(self) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(self.config.description_proj_pdrop),
            get_activation(self.config.description_proj_act),
            nn.Linear(self.text_encoder.config.hidden_size, self.language_model.config.hidden_size),
        )

    def set_type_loss(self) -> nn.CrossEntropyLoss:
        if self.config.type_weight is not None:
            self.type_loss = nn.CrossEntropyLoss(weight=torch.tensor(self.config.type_weight))
        else:
            self.type_loss = nn.CrossEntropyLoss()
        return self.type_loss

    def initialize_modules(
        self,
        text_encoder_path: str,
        language_model_path: str,
        phrase_encoder_path: str,
        text_encoder_proj_path: str | None = None,
        phrase_encoder_proj_path: str | None = None,
        **kwargs
    ):
        """Initialize the DVAModel with pre-trained language model and phrase encoder.

        If the `phrase_encoder_proj_path` is not provided, the projection layer will be initialized from scratch.
        :param language_model_path: The path to the pre-trained language model.
        :param phrase_encoder_path: The path to the pre-trained phrase encoder.
        :param phrase_encoder_proj_path: The path to the pre-trained phrase encoder projection layer.
        :param kwargs: Remaining keyword arguments.
        :return: None
        """
        self.text_encoder = AutoModel.from_pretrained(text_encoder_path, **kwargs)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_path, **kwargs)
        self.phrase_encoder = AutoModel.from_pretrained(phrase_encoder_path, **kwargs)
        self.sv_input_embeddings = self._build_sv_embeddings(self.language_model.get_input_embeddings())
        self.sv_output_embeddings = self._build_sv_embeddings(self.language_model.get_output_embeddings())
        self.config.text_encoder_config = self.text_encoder.config
        self.config.language_model_config = self.language_model.config
        self.config.phrase_encoder_config = self.phrase_encoder.config
        if self.config.use_text_encoder_proj:
            self.text_encoder_proj = self._build_text_encoder_proj()
            if text_encoder_proj_path is not None:
                if text_encoder_proj_path.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    state_dict = load_file(text_encoder_proj_path)
                else:
                    state_dict = torch.load(text_encoder_proj_path, map_location="cpu")

                self.text_encoder_proj.load_state_dict(state_dict, strict=False)
        if self.config.use_phrase_encoder_proj:
            self.phrase_encoder_proj = self._build_phrase_encoder_proj()
            if phrase_encoder_proj_path is not None:
                if phrase_encoder_proj_path.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    state_dict = load_file(phrase_encoder_proj_path)
                else:
                    state_dict = torch.load(phrase_encoder_proj_path, map_location="cpu")

                self.phrase_encoder_proj.load_state_dict(state_dict, strict=False)

    def get_text_embeddings(self, text_ids: torch.Tensor, text_attention_mask: torch.Tensor) -> torch.Tensor:
        text_embeddings = self.text_encoder(input_ids=text_ids, attention_mask=text_attention_mask)
        text_embeddings = text_embeddings.last_hidden_state
        if self.config.use_text_encoder_proj:
            text_embeddings = self.text_encoder_proj(text_embeddings)
        return text_embeddings

    def _get_phrase_embeddings(self, phrase_ids: torch.Tensor, phrase_attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract phrase embeddings from the phrase encoder.

        B: Batch size
        L: Sequence length
        D: Embedding dimension
        :param phrase_ids: Input ids of the phrases tokenized by the phrase tokenizer. (Shape: [B, L])
        :param phrase_attention_mask: Attention mask for the input ids. (Shape: [B, L])
        :return: Phrase embeddings of the input phrases. (Shape: [B, D])
        """
        outputs = self.phrase_encoder(input_ids=phrase_ids, attention_mask=phrase_attention_mask)
        phrase_embeddings = outputs.last_hidden_state
        phrase_end = torch.sum(phrase_attention_mask, dim=1)
        if self.config.use_phrase_encoder_proj:
            phrase_embeddings = self.phrase_encoder_proj(phrase_embeddings)

        phrase_embeddings = phrase_embeddings[range(len(phrase_embeddings)), phrase_end - 1]

        return phrase_embeddings

    @staticmethod
    def _filter_forward_params(
        forward_params: dict[str, Any], forward_func: Callable, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Update the forward method's parameters to match the target forward method's signature.

        :param forward_params: The parameters to be passed to the forward method.
        :param forward_func: The target forward function.
        :param kwargs: Additional keyword arguments to be passed to the forward method.
        """
        target_params = inspect.signature(forward_func).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in target_params.values()):
            # Check whether the language model's forward method accepts keyword arguments.
            forward_params.update(kwargs)
        else:
            # If not, filter out any unsupported parameters before calling it.
            forward_params = {k: v for k, v in forward_params.items() if k in target_params}

        return forward_params

    def get_dva_embeddings(
        self, phrase_ids: torch.Tensor | None, phrase_attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the input embeddings and output embeddings (LM Head) of the DVAModel.

        M: The size of the dynamic vocabulary (DV), i.e., the size of the phrase candidates.
        V: The size of the static vocabulary (SV), i.e., the vocab size of the language model.
        L: Sequence length
        :param phrase_ids: Input ids of the phrases tokenized by the phrase tokenizer. (Shape: [M, L])
        :param phrase_attention_mask: Attention mask for the input ids. (Shape: [M, L])
        :return: A tuple containing the input and output embeddings of the DVAModel. (Shape: [V+M, D])
        """
        if phrase_ids is None or phrase_attention_mask is None:
            # If the dynamic vocabulary is not provided (i.e., without phrase candidates),
            # the static vocabulary embeddings are returned as dva embeddings.
            return self.sv_input_embeddings, self.sv_output_embeddings

        dv_embeddings = [
            self._get_phrase_embeddings(
                phrase_ids[i : i + self.config.phrase_encoder_batch_size],
                phrase_attention_mask[i : i + self.config.phrase_encoder_batch_size],
            )
            for i in range(0, len(phrase_ids), self.config.phrase_encoder_batch_size)
        ]
        dva_input_embeddings = torch.cat([self.sv_input_embeddings, *dv_embeddings], dim=0)
        dva_output_embeddings = torch.cat([self.sv_output_embeddings, *dv_embeddings], dim=0)

        return dva_input_embeddings, dva_output_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        labels: torch.Tensor | None = None,
        type_labels: torch.Tensor | None = None,
        description_labels: torch.Tensor | None = None,
        description_attention_mask: torch.Tensor | None = None,
        text_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        text_embeds: torch.FloatTensor | None = None,
        phrase_ids: torch.Tensor | None = None,
        phrase_attention_mask: torch.Tensor | None = None,
        dva_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        """Forward function of the DVAModel.

        B: Batch size
        L: Sequence length
        P: Phrase length
        V: The size of the static vocabulary (SV), i.e., the vocab size of the language model.
        M: The size of the dynamic vocabulary (DV), i.e., the size of the phrase candidates.
        :param input_ids: Mixed input ids, including both token ids and phrase ids tokenized by DVATokenizer.
                          Indices should be in `[0, ..., V+M]` (Shape: [B, L])
        :param attention_mask: Attention mask for the input ids. (Shape: [B, L])
        :param labels: Labels for computing the causal language modeling loss.
                       Indices should either be in `[0, ..., V+M]` or `-100`. Tokens with indices set to `-100` are
                       ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., V+M]`.
        :param phrase_ids: Input ids of the phrases used to compute the dynamic vocabulary embeddings. (Shape: [M, P])
        :param phrase_attention_mask: Attention mask for the phrase ids. (Shape: [M, P])
        :param dva_embeds: Instead of passing `phrase_ids` and `phrase_attention_mask`, an alternative is to
                           pass the pre-computed input and output embeddings (SV + DV) of the DVAModel.
                           (A tuple consisting of `dva_input_embeddings` and `dva_output_embeddings`. Shape: [V+M, D])
        :return: `CausalLMOutputWithPast` containing the loss, logits, past key values, hidden states, and attentions.
        """
        if text_ids is not None and text_attention_mask is not None:
            text_embeds = self.get_text_embeddings(text_ids, text_attention_mask)
        else:
            assert text_embeds is not None, (
                "Either `text_ids` and `text_attention_mask` or `text_embeds` must be provided to compute the "
                "text embeddings."
            )
        if phrase_ids is not None and phrase_attention_mask is not None:
            dva_input_embeddings, dva_output_embeddings = self.get_dva_embeddings(phrase_ids, phrase_attention_mask)
        else:
            assert dva_embeds is not None, (
                "Either `phrase_ids` and `phrase_attention_mask` or `dva_embeds` must be provided to compute the "
                "DVA input and output embeddings."
            )
            dva_input_embeddings, dva_output_embeddings = dva_embeds

        # Get the DV embeddings for auxiliary losses.
        phrase_embeds = dva_input_embeddings[self.vocab_size:]

        if inputs_embeds is None:
            inputs_embeds = dva_input_embeddings[input_ids]

        multimodal_inputs_embeds = torch.cat([text_embeds, inputs_embeds], dim=1)
        multimodal_attention_mask = torch.cat([text_attention_mask, attention_mask], dim=1)
        multimodal_position_ids = multimodal_attention_mask.long().cumsum(-1) - 1
        multimodal_position_ids.masked_fill_(multimodal_attention_mask == 0, 1)

        forward_params = {
            "inputs_embeds": multimodal_inputs_embeds,
            "attention_mask": multimodal_attention_mask,
            "position_ids": multimodal_position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "cache_position": cache_position,
        }
        forward_params = self._filter_forward_params(forward_params, self.language_model.base_model.forward, kwargs)
        # print(f"{text_embeds.shape=}", f"{inputs_embeds.shape=}")
        # print(f"{text_attention_mask.shape=}", f"{attention_mask.shape=}")
        # print(f"{forward_params['inputs_embeds'].shape=}", f"{forward_params['attention_mask'].shape=}")
        outputs = self.language_model.base_model(**forward_params)
        hidden_states = outputs.last_hidden_state
        logits = hidden_states @ dva_output_embeddings.T

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=dva_output_embeddings.shape[0], **kwargs
            )
            if self.config.use_type_loss and type_labels is not None:
                type_logits = self.type_classification_head(phrase_embeds)
                type_loss = self.type_loss(type_logits, type_labels)
                loss += self.type_loss_weight * type_loss
            if self.config.use_description_loss and description_labels is not None:
                description_reps = self.text_encoder.base_model(
                    input_ids=description_labels,
                    attention_mask=description_attention_mask,
                )
                description_hidden_states = description_reps.last_hidden_state
                # We apply mean pooling and projection to get a single representation for each description.
                mask = description_attention_mask.float().unsqueeze(-1)
                masked_description_reps = description_hidden_states * mask
                mean_description_reps = masked_description_reps.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_description_reps = self.description_proj(mean_description_reps)
                description_loss = self.description_loss(mean_description_reps, phrase_embeds)
                loss += self.description_loss_weight * description_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# This class is used to mask the logits for batch inference.
class DVALogitsProcessor(LogitsProcessor):
    def __init__(self, mask_phrase_ids: list[list[int]]):
        super().__init__()
        self.mask_phrase_ids = mask_phrase_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert len(self.mask_phrase_ids) == scores.size(0), (
            f"Mask token ids size {len(self.mask_phrase_ids)} and scores batch size {scores.size(0)} mismatch."
        )

        for i, mask_ids in enumerate(self.mask_phrase_ids):
            if mask_ids:
                scores[i].index_fill_(0, torch.tensor(mask_ids, device=scores.device), float("-inf"))
        return scores
