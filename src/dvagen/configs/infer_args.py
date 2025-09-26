from dataclasses import dataclass


@dataclass
class InferArguments:
    doc_top_k: int
    embedding_model_path: str
    data_file: str | None = None
    vector_store_path: str | None = None
    save_vector_store_path: str | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    max_length: int | None = None
    max_new_tokens: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    protein_sequence_mapping_file: str | None = None
