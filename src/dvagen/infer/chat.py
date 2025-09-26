from ..configs.parser import InferArgs
from .infer import infer, prepare


def chat(infer_args: InferArgs):
    model, phrase_sampler, tokenizer, retriever = prepare(
        dva_model_path=infer_args.model.model_name_or_path,
        retriever_embedding_model_path=infer_args.infer.embedding_model_path,
        text_tokenizer_path=infer_args.model.text_encoder_path,
        phrase_encoder_batch_size=infer_args.model.phrase_encoder_batch_size,
        lm_tokenizer_path=infer_args.model.language_model_path,
        phrase_tokenizer_path=infer_args.model.phrase_encoder_path,
        retriever_data_file=infer_args.infer.data_file,
        retriever_vector_store_path=infer_args.infer.vector_store_path,
        retriever_save_vector_store_path=infer_args.infer.save_vector_store_path,
        phrase_sampler_type=infer_args.model.phrase_sampler_type,
        sampler_model_path=infer_args.model.sampler_model_path,
        sampler_random_up=infer_args.model.sampler_random_up,
        sampler_random_low=infer_args.model.sampler_random_low,
        phrase_max_length=infer_args.model.phrase_max_length,
        fmm_embedding_model_path=infer_args.model.fmm_embedding_model_path,
        fmm_data_file=infer_args.model.fmm_data_file,
        fmm_vector_store_path=infer_args.model.fmm_vector_store_path,
        fmm_save_vector_store_path=infer_args.model.fmm_save_vector_store_path,
        fmm_min_length=infer_args.model.fmm_min_length,
        fmm_max_length=infer_args.model.fmm_max_length,
        protein_fragment_mapping_file=infer_args.model.protein_fragment_mapping_file,
    )

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        outputs = infer(
            model,
            phrase_sampler,
            tokenizer,
            retriever,
            queries=[query],
            doc_top_k=infer_args.infer.doc_top_k,
            do_sample=infer_args.infer.do_sample,
            max_new_tokens=infer_args.infer.max_new_tokens,
            temperature=infer_args.infer.temperature,
            top_k=infer_args.infer.top_k,
            protein_sequence_mapping_file=infer_args.infer.protein_sequence_mapping_file,
        )
        print("Assistant: ", end="", flush=True)
        print(outputs[0]["decoded_sentence"])
