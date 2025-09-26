from dataclasses import dataclass


@dataclass
class Document:
    content: str = None
    token_ids: list[int] = None
    id: int = None


@dataclass
class Phrase:
    content: str
    is_phrase: bool
    src_doc_id: int | None = None  # ID of the document this phrase comes from
    type: int | None = None  # Special field for ProDVa
    description: str | None = None  # Special field for ProDVa
