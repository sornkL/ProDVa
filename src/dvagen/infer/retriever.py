import json

from abc import ABC, abstractmethod

import faiss
import langchain_core.documents
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ..models.phrase import Document
from ..utils import logging


logger = logging.get_logger(__name__)


class BaseRetriever(ABC):
    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def retrieve_documents(self, query: str, top_k: int) -> list[Document]: ...


class RandomRetriever(BaseRetriever):
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.documents = []
        with open(self.data_file) as file:
            data = json.load(file)
        for item in data:
            self.documents.append(item["instruction"])

    def retrieve_documents(self, query: str, top_k: int) -> list[Document]:
        import random

        random_documents = random.sample(self.documents, top_k)
        return [Document(content=doc) for doc in random_documents]


class FAISSRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_model_path: str,
        data_file: str = None,
        vector_store_path: str = None,
        save_vector_store_path: str = None,
    ):
        self.embedding_model_path = embedding_model_path
        self.data_file = data_file
        self.vector_store_path = vector_store_path
        self.use_cuda = torch.cuda.is_available()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={"device": "cuda" if self.use_cuda else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if vector_store_path is not None:
            # Load the FAISS index from disk
            if self.data_file is not None:
                logger.warning_rank0(
                    "Both `data_file` and `vector_store_path` are provided. Using `vector_store_path` to load the "
                    "FAISS index. Ignoring `data_file`."
                )
            self.vector_store = FAISS.load_local(
                self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True
            )
        else:
            assert self.data_file is not None, "`data_file` is required to build a vector store."

            self.documents = []
            with open(self.data_file) as file:
                data = json.load(file)
            for item in data:
                self.documents.append(item["instruction"])
            self.documents = [
                langchain_core.documents.Document(document, metadata={"id": idx})
                for idx, document in enumerate(self.documents)
            ]
            self.vector_store = FAISS.from_documents(self.documents, self.embedding_model)
            if save_vector_store_path is not None:
                self.vector_store.save_local(save_vector_store_path)
                logger.info_rank0(f"Saved FAISS vector store to {save_vector_store_path}")

        if self.use_cuda:
            self.vector_store.index = faiss.index_cpu_to_all_gpus(self.vector_store.index)

    def retrieve_documents(self, query: str, top_k: int) -> list[Document]:
        top_k_documents = self.vector_store.similarity_search(query, k=top_k)
        return [
            Document(content=document.page_content, id=document.metadata.get("id")) for document in top_k_documents
        ]
