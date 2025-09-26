import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer

from ..configs.model_args import PhraseSamplerType
from ..configs.parser import TrainArgs
from ..configs.train_args import FinetuningType
from ..datasets.dvadataset import DVADataset
from ..models.configuration_dva import DVAConfig
from ..models.modeling_dva import DVAModel
from ..models.sampler import FMMPhraseSampler, NTokenPhraseSampler, NWordsPhraseSampler, ProteinFragmentSampler
from ..models.tokenization_dva import DVATokenizer
from ..utils import logging
from .trainer import DVATrainer


logger = logging.get_logger(__name__)


class DVACollator:
    def __init__(self, tokenizer: DVATokenizer, max_text_length: int, device, params=None):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.device = device

    def __call__(self, batch):
        batch_instruction = [item[0] for item in batch]
        batch_inputs = [item[1] for item in batch]

        batch_types = []
        batch_descriptions = []
        for phrases in batch_inputs:
            if len(phrases) == 0:
                continue
            for phrase in phrases:
                if phrase.is_phrase:
                    batch_types.append(phrase.type)
                    batch_descriptions.append(phrase.description)

        inputs = self.tokenizer.batch_encode(batch_inputs, phrases_mask=False)
        text_inputs = self.tokenizer.text_tokenizer(
            batch_instruction,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_text_length,
        )
        if len(batch_descriptions) > 0:
            description_inputs = self.tokenizer.text_tokenizer(
                batch_descriptions,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_text_length,
            )
            inputs["description_labels"] = description_inputs["input_ids"]
            inputs["description_attention_mask"] = description_inputs["attention_mask"]
        else:
            inputs["description_labels"] = None
            inputs["description_attention_mask"] = None
        inputs["text_ids"] = text_inputs["input_ids"]
        inputs["text_attention_mask"] = text_inputs["attention_mask"]
        text_labels = torch.full(text_inputs["input_ids"].shape, -100, dtype=torch.long)
        labels = torch.where(inputs["attention_mask"] == 1, inputs["input_ids"], torch.tensor(-100))
        inputs["labels"] = torch.cat([text_labels, labels], dim=1)
        if len(batch_types) > 0:
            inputs["type_labels"] = torch.tensor(batch_types, dtype=torch.long)
        else:
            inputs["type_labels"] = None

        return inputs


def train(train_args: TrainArgs):
    model_config = DVAConfig(
        text_encoder_config=AutoConfig.from_pretrained(train_args.model.text_encoder_path),
        language_model_config=AutoConfig.from_pretrained(train_args.model.language_model_path),
        phrase_encoder_config=AutoConfig.from_pretrained(train_args.model.phrase_encoder_path),
        use_text_encoder_proj=train_args.model.use_text_encoder_proj,
        text_encoder_proj_pdrop=train_args.model.text_encoder_proj_pdrop,
        text_encoder_proj_act=train_args.model.text_encoder_proj_act,
        use_phrase_encoder_proj=train_args.model.use_phrase_encoder_proj,
        phrase_encoder_proj_pdrop=train_args.model.phrase_encoder_proj_pdrop,
        phrase_encoder_proj_act=train_args.model.phrase_encoder_proj_act,
        phrase_encoder_batch_size=train_args.model.phrase_encoder_batch_size,
        use_type_loss=train_args.model.use_type_loss,
        use_description_loss=train_args.model.use_description_loss,
        type_loss_weight=train_args.model.type_loss_weight,
        description_loss_weight=train_args.model.description_loss_weight,
        type_classification_head_pdrop=train_args.model.type_classification_head_pdrop,
        description_proj_pdrop=train_args.model.description_proj_pdrop,
        type_classification_head_act=train_args.model.type_classification_head_act,
        description_proj_act=train_args.model.description_proj_act,
        type_classification_head_num_classes=train_args.model.type_classification_head_num_classes,
    )

    model = DVAModel(model_config)
    model.initialize_modules(
        text_encoder_path=train_args.model.text_encoder_path,
        language_model_path=train_args.model.language_model_path,
        phrase_encoder_path=train_args.model.phrase_encoder_path,
    )
    if train_args.train.finetuning_type == FinetuningType.LORA.value:
        assert train_args.train.lora is not None, "LoRA arguments must be provided for LoRA fine-tuning."
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_args.train.lora.r,
            lora_alpha=train_args.train.lora.alpha,
            lora_dropout=train_args.train.lora.dropout,
            target_modules=train_args.train.lora.target_modules,
        )
        model = get_peft_model(model, peft_config)
    elif train_args.train.finetuning_type == FinetuningType.FULL.value:
        if train_args.train.lora is not None:
            logger.warning_rank0(
                "LoRA arguments are provided but finetuning type is set to 'full'. Ignoring LoRA settings."
            )
    elif train_args.train.finetuning_type == FinetuningType.FREEZE.value:
        if train_args.train.freeze_text_encoder:
            logger.info_rank0("The text encoder is frozen during training.")
            for param in model.text_encoder.parameters():
                param.requires_grad_(False)
        if train_args.train.freeze_language_model:
            logger.info_rank0("The language model is frozen during training.")
            for param in model.language_model.parameters():
                param.requires_grad_(False)
    else:
        raise ValueError(f"Unsupported finetuning type: {train_args.train.finetuning_type}")

    # TODO In the future version, PhraseSamplerConfig will be used instead.
    if train_args.model.phrase_sampler_type == PhraseSamplerType.N_TOKENS:
        phrase_tokenizer = AutoTokenizer.from_pretrained(train_args.model.sampler_model_path)
        sampler = NTokenPhraseSampler(
            tokenizer=phrase_tokenizer,
            random_up=train_args.model.sampler_random_up,
            random_low=train_args.model.sampler_random_low,
            phrase_max_length=train_args.model.phrase_max_length,
        )
    elif train_args.model.phrase_sampler_type == PhraseSamplerType.N_WORDS:
        sampler = NWordsPhraseSampler(
            random_up=train_args.model.sampler_random_up,
            random_low=train_args.model.sampler_random_low,
            phrase_max_length=train_args.model.phrase_max_length,
        )
    elif train_args.model.phrase_sampler_type == PhraseSamplerType.FMM:
        sampler = FMMPhraseSampler(
            ignore_first=train_args.model.ignore_first,
            embedding_model_path=train_args.model.fmm_embedding_model_path,
            data_file=train_args.model.fmm_data_file,
            vector_store_path=train_args.model.fmm_vector_store_path,
            min_length=train_args.model.fmm_min_length,
            max_length=train_args.model.fmm_max_length,
        )
    elif train_args.model.phrase_sampler_type == PhraseSamplerType.PROTEIN_FRAGMENT:
        sampler = ProteinFragmentSampler(
            mapping_file=train_args.model.protein_fragment_mapping_file,
            format_sequence=True,
        )

    tokenizer = DVATokenizer(
        text_encoder_name_or_path=train_args.model.text_encoder_path,
        model_name_or_path=train_args.model.language_model_path,
        phrase_encoder_name_or_path=train_args.model.phrase_encoder_path,
        phrase_sampler_type=train_args.model.phrase_sampler_type,
        static_vocab=model.vocab_size,
        sampler=sampler,
    )
    training_set = DVADataset(
        tokenizer=tokenizer,
        protein_fragment_mapping_file=train_args.model.protein_fragment_mapping_file,
        data_path=train_args.data.train_path,
        save_data_path=train_args.data.save_train_path,
        max_sequence_length=train_args.data.max_sequence_length,
        max_phrase_length=train_args.data.max_phrase_length,
    )
    # If `use_type_loss` is True, we first compute the type weights and re-initialize the model's type loss.
    if train_args.model.use_type_loss:
        type_weight = training_set.get_type_weight(verbose=True)
        model.config.type_weight = type_weight
        model.set_type_loss()
    collator = DVACollator(
        tokenizer=tokenizer,
        max_text_length=train_args.data.max_text_length,
        device=train_args.train.device
    )
    # TODO support custom scheduler and optimizer
    trainer = DVATrainer(
        model=model.cuda(),
        args=train_args.train,
        train_dataset=training_set,
        data_collator=collator,
    )
    if train_args.train.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=train_args.train.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
