import json
import os.path

import torch
from transformers import AutoTokenizer, LogitsProcessorList

from ..configs.model_args import PhraseSamplerType
from ..models.modeling_dva import DVALogitsProcessor, DVAModel
from ..models.phrase import Document
from ..models.sampler import (
    BasePhraseSampler,
    FMMPhraseSampler,
    NTokenPhraseSampler,
    NWordsPhraseSampler,
    ProteinFragmentSampler,
)
from ..models.tokenization_dva import DVATokenizer
from ..utils.visualization import get_visualization
from .retriever import BaseRetriever, FAISSRetriever


def prepare(
    dva_model_path: str,
    retriever_embedding_model_path: str,
    phrase_encoder_batch_size: int = 64,
    text_tokenizer_path: str = None,
    lm_tokenizer_path: str = None,
    phrase_tokenizer_path: str = None,
    retriever_data_file: str = None,
    retriever_vector_store_path: str = None,
    retriever_save_vector_store_path: str = None,
    phrase_sampler_type: PhraseSamplerType = PhraseSamplerType.N_TOKENS,
    sampler_model_path: str = None,
    sampler_random_up: int = None,
    sampler_random_low: int = None,
    phrase_max_length: int = None,
    fmm_embedding_model_path: str = None,
    fmm_data_file: str = None,
    fmm_vector_store_path: str = None,
    fmm_save_vector_store_path: str = None,
    fmm_min_length: int = 2,
    fmm_max_length: int = 16,
    protein_fragment_mapping_file: str = None,
) -> tuple:
    if text_tokenizer_path is None:
        text_tokenizer_path = os.path.join(dva_model_path, "text_tokenizer")
    if lm_tokenizer_path is None:
        lm_tokenizer_path = os.path.join(dva_model_path, "lm_tokenizer")
    if phrase_tokenizer_path is None:
        phrase_tokenizer_path = os.path.join(dva_model_path, "phrase_tokenizer")

    # DVAModel
    model = DVAModel.from_pretrained(
        dva_model_path, device_map="auto", phrase_encoder_batch_size=phrase_encoder_batch_size
    )
    model.eval()

    # Phrase Sampler
    if phrase_sampler_type == PhraseSamplerType.N_TOKENS:
        phrase_tokenizer = AutoTokenizer.from_pretrained(sampler_model_path)
        phrase_sampler = NTokenPhraseSampler(
            tokenizer=phrase_tokenizer,
            random_up=sampler_random_up,
            random_low=sampler_random_low,
            phrase_max_length=phrase_max_length,
        )
    elif phrase_sampler_type == PhraseSamplerType.N_WORDS:
        phrase_sampler = NWordsPhraseSampler(
            random_up=sampler_random_up,
            random_low=sampler_random_low,
            phrase_max_length=phrase_max_length,
        )
    elif phrase_sampler_type == PhraseSamplerType.FMM:
        phrase_sampler = FMMPhraseSampler(
            ignore_first=True,
            embedding_model_path=fmm_embedding_model_path,
            data_file=fmm_data_file,
            vector_store_path=fmm_vector_store_path,
            save_vector_store_path=fmm_save_vector_store_path,
            min_length=fmm_min_length,
            max_length=fmm_max_length,
        )
    elif phrase_sampler_type == PhraseSamplerType.PROTEIN_FRAGMENT:
        phrase_sampler = ProteinFragmentSampler(
            mapping_file=protein_fragment_mapping_file,
            format_sequence=False,
        )

    # Tokenizer
    tokenizer = DVATokenizer(
        text_encoder_name_or_path=text_tokenizer_path,
        model_name_or_path=lm_tokenizer_path,
        phrase_encoder_name_or_path=phrase_tokenizer_path,
        static_vocab=model.config.language_model_config.vocab_size,
        sampler=phrase_sampler,
    )
    tokenizer.lm_tokenizer.padding_side = "left"  # We set the padding side to left during inference

    # Retriever
    retriever = FAISSRetriever(
        embedding_model_path=retriever_embedding_model_path,
        data_file=retriever_data_file,
        vector_store_path=retriever_vector_store_path,
        save_vector_store_path=retriever_save_vector_store_path,
    )

    return model, phrase_sampler, tokenizer, retriever


@torch.no_grad()
def infer(
    model: DVAModel,
    phrase_sampler: BasePhraseSampler,
    tokenizer: DVATokenizer,
    retriever: BaseRetriever,
    queries: list[str],
    doc_top_k: int,
    protein_sequence_mapping_file: str = None,
    return_ids: bool = False,
    visualize: bool = False,
    **kwargs,
):
    with open(protein_sequence_mapping_file) as f:
        sequence_mappings = json.load(f)
    sequence_mappings = {item["instruction"]: item["sequence"] for item in sequence_mappings}

    supporting_documents_list = [retriever.retrieve_documents(query, doc_top_k) for query in queries]
    supporting_sequences_list = [
        [Document(content=sequence_mappings[doc.content], id=doc.id) for doc in documents]
        for documents in supporting_documents_list
    ]
    phrase_candidates_list = [
        [phrase for document in documents for phrase in phrase_sampler.sample(document)]
        for documents in supporting_sequences_list
    ]
    phrase_inputs = tokenizer.batch_encode(phrase_candidates_list, phrases_mask=True)
    text_prefix_inputs = tokenizer.text_tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512)

    sequence_prefix_inputs = tokenizer.lm_tokenizer([
        tokenizer.lm_tokenizer.eos_token + "\n" for _ in range(len(queries))
    ], return_tensors="pt", padding=True, truncation=True, max_length=512)

    text_ids = text_prefix_inputs["input_ids"].to(model.device)
    text_attention_mask = text_prefix_inputs["attention_mask"].to(model.device)
    input_ids = sequence_prefix_inputs["input_ids"].to(model.device)
    attention_mask = sequence_prefix_inputs["attention_mask"].to(model.device)
    phrase_ids = phrase_attention_mask = None
    if len(phrase_inputs["phrase_ids"]):
        phrase_ids = phrase_inputs["phrase_ids"].to(model.device)
        phrase_attention_mask = phrase_inputs["phrase_attention_mask"].to(model.device)
    mask_phrase_ids = phrase_inputs["mask_ids"]

    text_embeds = model.get_text_embeddings(text_ids, text_attention_mask)
    dva_embeds = model.get_dva_embeddings(phrase_ids, phrase_attention_mask)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        text_embeds=text_embeds,
        text_attention_mask=text_attention_mask,
        dva_embeds=dva_embeds,
        use_cache=False,
        logits_processor=LogitsProcessorList([DVALogitsProcessor(mask_phrase_ids)]),
        output_scores=visualize,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.lm_tokenizer.eos_token_id,
        pad_token_id=tokenizer.lm_tokenizer.pad_token_id,
        **kwargs,
    )
    if phrase_ids is not None:
        phrase_ids = phrase_ids.tolist()
    res = [tokenizer.decode(output.tolist(), phrase_ids, return_ids=return_ids) for output in outputs.sequences]

    if visualize:
        for idx in range(len(res)):
            suffix_ids = outputs.sequences[idx][len(input_ids[idx]) :]
            suffix_scores = outputs.scores

            tmp = []
            for step in range(len(suffix_ids)):
                step_id = suffix_ids[step].item()
                step_token = tokenizer.lm_tokenizer.decode([step_id])
                step_logits = suffix_scores[step][idx]
                step_prob = torch.softmax(step_logits, dim=-1)

                tmp.append(
                    {
                        "token": step_token,
                        "type": "token" if step_id < tokenizer.static_vocab else "phrase",
                        "prob": step_prob[int(step_id)].item(),
                    }
                )

            res[idx].update({"visualization": get_visualization(tmp, **kwargs)})
    return res
