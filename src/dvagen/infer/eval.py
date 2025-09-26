import json
import os

from tqdm.auto import tqdm
from transformers import set_seed

from ..configs.eval_args import EvalTaskType
from ..configs.parser import EvalArgs
from ..infer.infer import infer, prepare
from ..utils import logging


logger = logging.get_logger(__name__)


def predict_results(
    eval_args: EvalArgs,
) -> list[dict[str, str]]:
    model, phrase_sampler, tokenizer, retriever = prepare(
        dva_model_path=eval_args.model.model_name_or_path,
        retriever_embedding_model_path=eval_args.infer.embedding_model_path,
        text_tokenizer_path=eval_args.model.text_encoder_path,
        phrase_encoder_batch_size=eval_args.model.phrase_encoder_batch_size,
        lm_tokenizer_path=eval_args.model.language_model_path,
        phrase_tokenizer_path=eval_args.model.phrase_encoder_path,
        retriever_data_file=eval_args.infer.data_file,
        retriever_vector_store_path=eval_args.infer.vector_store_path,
        retriever_save_vector_store_path=eval_args.infer.save_vector_store_path,
        phrase_sampler_type=eval_args.model.phrase_sampler_type,
        sampler_model_path=eval_args.model.sampler_model_path,
        sampler_random_up=eval_args.model.sampler_random_up,
        sampler_random_low=eval_args.model.sampler_random_low,
        phrase_max_length=eval_args.model.phrase_max_length,
        fmm_embedding_model_path=eval_args.model.fmm_embedding_model_path,
        fmm_data_file=eval_args.model.fmm_data_file,
        fmm_vector_store_path=eval_args.model.fmm_vector_store_path,
        fmm_save_vector_store_path=eval_args.model.fmm_save_vector_store_path,
        fmm_min_length=eval_args.model.fmm_min_length,
        fmm_max_length=eval_args.model.fmm_max_length,
        protein_fragment_mapping_file=eval_args.model.protein_fragment_mapping_file,
    )

    with open(eval_args.eval.test_data_file) as f:
        test_data = json.load(f)

    references = []
    predictions = []
    instructions = []
    for query in tqdm(range(0, len(test_data), eval_args.eval.batch_size)):
        batch_queries = [q["instruction"].strip() for q in test_data[query : query + eval_args.eval.batch_size]]
        batch_references = [q["sequence"].strip() for q in test_data[query : query + eval_args.eval.batch_size]]
        references.extend(batch_references)
        instructions.extend(batch_queries)
        batch_outputs = infer(
            model,
            phrase_sampler,
            tokenizer,
            retriever,
            return_ids=True,
            queries=batch_queries,
            doc_top_k=eval_args.infer.doc_top_k,
            do_sample=eval_args.infer.do_sample,
            temperature=eval_args.infer.temperature,
            top_k=eval_args.infer.top_k,
            max_new_tokens=eval_args.infer.max_new_tokens,
            protein_sequence_mapping_file=eval_args.infer.protein_sequence_mapping_file,
        )
        predictions.extend(batch_outputs)

    predicted_results = [
        {"instruction": inst, "prediction": pred["decoded_sentence"], "reference": ref, "ids": pred["ids"]}
        for inst, pred, ref in zip(instructions, predictions, references)
    ]

    if eval_args.eval.save_results_path is not None:
        with open(eval_args.eval.save_results_path, "w") as f:
            json.dump(predicted_results, f)

    return predicted_results


def report_metrics(predicted_results: list[dict[str, str | list]], eval_args: EvalArgs) -> dict[str, float]:
    predictions = [result["prediction"] for result in predicted_results]
    references = [result["reference"] for result in predicted_results]
    predictions_ids = [result["ids"] for result in predicted_results]

    if eval_args.eval.task_type == EvalTaskType.LANGUAGE_MODELING:
        logger.info_rank0(f"ProDVa does not support {eval_args.eval.task_type.value} task.")
        return None
    elif eval_args.eval.task_type == EvalTaskType.PROTEIN_DESIGN:
        logger.info_rank0(
            "Evaluation on protein design metrics will be supported in a future version. "
            "For now, please refer to https://github.com/PDFBench/PDFBench."
        )
        return None
    else:
        raise ValueError(f"Unsupported eval task type: {eval_args.eval.task_type}")

    # return eval_metrics.compute()


def evaluate(eval_args: EvalArgs):
    if eval_args.eval.eval_seed is not None:
        set_seed(eval_args.eval.eval_seed)

    if eval_args.eval.save_results_path is not None and os.path.exists(eval_args.eval.save_results_path):
        with open(eval_args.eval.save_results_path) as f:
            predicted_results = json.load(f)
        return report_metrics(predicted_results, eval_args)

    predicted_results = predict_results(eval_args)
    return report_metrics(predicted_results, eval_args)
