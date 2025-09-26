import json
import random
import re
from abc import ABC, abstractmethod

import nltk

from ..infer.retriever import FAISSRetriever
from .phrase import Document, Phrase


class BasePhraseSampler(ABC):
    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def sample(self, document: Document) -> list[Phrase]: ...


class NTokenPhraseSampler(BasePhraseSampler):
    def __init__(
        self,
        tokenizer,
        random_up=12,
        random_low=8,
        phrase_max_length=5,
    ):
        self.tokenizer = tokenizer
        self.random_up = random_up
        self.random_low = random_low
        self.phrase_max_length = phrase_max_length
        self.phrase_num = 0

    def sample(self, document: Document) -> list[Phrase]:
        phrases = []
        tokens = self.tokenizer.tokenize(document.content)
        end_index = len(tokens)
        now = 0
        while now < end_index:
            start = random.randint(now + self.random_low, now + self.random_up)
            end = start + random.randint(2, self.phrase_max_length)
            if end > end_index or start > end:
                content = self.tokenizer.convert_tokens_to_string(tokens[now:end_index])
                phrases.append(Phrase(content=content, is_phrase=False))
                break
            content = self.tokenizer.convert_tokens_to_string(tokens[now:start])
            phrases.append(Phrase(content=content, is_phrase=False))

            phrases.append(
                Phrase(
                    content=self.tokenizer.convert_tokens_to_string(tokens[start:end]),
                    is_phrase=True,
                )
            )
            now = end
        return phrases

    def sample_negative(self, document: Document) -> list[Phrase]: ...


class NWordsPhraseSampler(BasePhraseSampler):
    def __init__(
        self,
        random_up=12,
        random_low=8,
        phrase_max_length=5,
    ):
        self.random_up = random_up
        self.random_low = random_low
        self.phrase_max_length = phrase_max_length
        self.phrase_num = 0

    def sample(self, document: Document) -> list[Phrase]:
        phrases = []
        words = document.content.split(" ")
        end_index = len(words)
        now = 0
        while now < end_index:
            start = random.randint(now + self.random_low, now + self.random_up)
            end = start + random.randint(2, self.phrase_max_length)
            if end > end_index or start > end:
                content = " ".join(words[now:end_index])
                phrases.append(Phrase(content=content, is_phrase=False))
                break
            content = " ".join(words[now:start])
            phrases.append(Phrase(content=content, is_phrase=False))

            phrases.append(
                Phrase(
                    content=" ".join(words[start:end]),
                    is_phrase=True,
                )
            )
            now = end
        return phrases


class FMMPhraseSampler(BasePhraseSampler):
    # ------------------------subclass SearchItem------------------------#
    class SearchItem:
        def __init__(
            self,
            min_length,
            max_length,
            text,
            ignore_first=False,
            docs=None,
            punc=None,
        ):
            self.text = text
            self.ignore_first = ignore_first
            self.docs = docs
            data, self.data_pos = nltk.word_tokenize(self.text), []
            # combine the '< |endoftext| >' to '<|endoftext|>'
            self.data = []
            for token in data:
                if token not in [">", "|endoftext|"]:
                    if token == "``":
                        self.data.append('"')
                    else:
                        self.data.append(token)
                else:
                    if len(self.data) > 0:
                        if self.data[-1] == "<" and token == "|endoftext|":
                            self.data[-1] += token
                        elif self.data[-1] == "<|endoftext|" and token == ">":
                            self.data[-1] += token
                        else:
                            self.data.append(token)

            # self.self_doc_index = item[0]
            # self.candidates = [i for i in list(item[1]) if i != item[0]]
            self.candidates = [i for i in range(len(docs))]
            if self.ignore_first:
                self.candidates = self.candidates[1:]
            self.min_length, self.max_length = min_length, max_length
            self.pointer = 0
            self.result = []
            self.punc = punc
            self.last_rest, self.current_rest = [], []
            self.index = -1
            self.cache = []

        def move(self):
            while not self.is_end():
                if len(self.cache) > 1 and self.cache[-1] in self.punc:
                    self.last_rest = self.current_rest
                    self.move_back()
                    self.save_once()
                elif len(self.cache) == 1 and self.cache[0] in self.punc:
                    self.save_once()
                elif self.get_length() > self.max_length:
                    self.current_rest = []
                elif self.min_length <= self.get_length() <= self.max_length:
                    self.search_now()

                if len(self.last_rest) > 0 and len(self.current_rest) == 0:
                    self.move_back()
                    self.save_once()
                elif len(self.last_rest) == 0 and len(self.current_rest) > 0:
                    pass
                elif len(self.last_rest) > 0 and len(self.current_rest) > 0:
                    pass
                elif len(self.last_rest) == 0 and len(self.current_rest) == 0:
                    if self.min_length <= self.get_length() <= self.max_length:
                        self.save_once()
                self.move_once()
            if len(self.last_rest) == 0 and len(self.current_rest) > 0:
                self.save_once()
            elif len(self.last_rest) > 0 and len(self.current_rest) > 0:
                self.save_once()
            elif len(self.last_rest) == 0 and len(self.current_rest) == 0:
                self.save_once()

        def search_now(self):
            string = " ".join(self.cache)
            index, docid = -1, -1

            self_doc = " ".join([i for i, j in self.result])
            self.candidates = [None] + self.candidates

            for did in self.candidates:
                if did:
                    doc = self.docs[did]
                else:
                    doc = self_doc
                try:
                    index = doc.index(string)
                except:
                    continue
                if index != -1:
                    docid = did
                    break
            if docid != -1 and index != -1:
                self.save_current_rest([(docid, index)])
            else:
                self.save_current_rest([])

        def get_length(self):
            return len(self.cache)

        def save_current_rest(self, rest):
            self.last_rest = self.current_rest
            self.current_rest = rest

        def save_once(self):
            self.result.append((" ".join(self.cache), self.current_rest))
            self.cache = []
            self.last_rest, self.current_rest = [], []

        def move_once(self):
            self.cache.append(self.data[self.pointer])
            self.pointer += 1

        def is_end(self):
            return False if self.pointer < len(self.data) else True

        def move_back(self):
            self.current_rest = self.last_rest
            self.last_rest = []
            self.cache = self.cache[:-1]
            # NOTE:
            if self.pointer < len(self.data):
                self.pointer -= 1

    # ------------------------subclass SearchItem------------------------#
    def __init__(
        self,
        ignore_first=False,
        embedding_model_path=None,
        data_file=None,
        vector_store_path=None,
        save_vector_store_path=None,
        min_length=2,
        max_length=16,
    ):
        self.ignore_first = ignore_first
        self.min_length = min_length
        self.max_length = max_length
        self.retriever = FAISSRetriever(
            embedding_model_path=embedding_model_path,
            data_file=data_file,
            vector_store_path=vector_store_path,
        )

    def clean_data(self, result):
        units = []
        empty_cache = []
        for unit in result:
            if unit[1]:
                if empty_cache:
                    units.append((" ".join(empty_cache), []))
                units.append(unit)
                empty_cache = []
            else:
                empty_cache.append(unit[0])
        if empty_cache:
            units.append((" ".join(empty_cache), []))
        return units

    def sample(self, document: Document, topk) -> list[Phrase]:
        docs = self.retriever.retrieve_documents(document.content, topk)
        docs = [doc.content for doc in docs]
        min_length, max_length = self.min_length, self.max_length
        punc = set(
            [
                ",",
                ".",
                '"',
                "'",
                "?",
                "!",
                "@",
                "-",
                "<",
                ">",
                ":",
                ";",
                "/",
                "_",
                "+",
                "=",
                "~",
                "`",
                "#",
                "$",
                "%",
                "^",
                "&",
                "*",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
            ]
        )
        searchitem = self.SearchItem(
            min_length, max_length, text=document.content, ignore_first=self.ignore_first, docs=docs, punc=punc
        )
        searchitem.move()
        results = {"results": self.clean_data(searchitem.result), "index": searchitem.index}
        phrases = []
        for _result in results["results"]:
            if _result[1]:
                phrases.append(
                    Phrase(
                        content=_result[0],
                        is_phrase=True,
                    )
                )
            else:
                phrases.append(
                    Phrase(
                        content=_result[0],
                        is_phrase=False,
                    )
                )
        return phrases


class ProteinFragmentSampler(BasePhraseSampler):
    def __init__(self, mapping_file: str, format_sequence: bool):
        with open(mapping_file) as f:
            self.fragment_mappings = json.load(f)

        # During training, if ProtGPT2 is used as the Protein Language Model backbone,
        # we need to format the amino acid sequence to the FASTA file format.
        # Otherwise, including during inference, we do not need to format the sequence to sample phrases.
        self.format_sequence = format_sequence

        self.fragment_mappings = {
            self.format_amino_acid_sequence(item["sequence"]) if self.format_sequence else item["sequence"]: item["phrases"] for item in self.fragment_mappings
        }

    def get_fragments(self, sequence: str) -> list[str]:
        fragments = self.fragment_mappings.get(sequence, [])
        if len(fragments) > 0:
            fragments = [fragment["phrase"] for fragment in fragments]
        return fragments

    @staticmethod
    def format_amino_acid_sequence(sequence: str) -> str:
        """Convert the amino acid sequence to the FASTA file format.
        Note that <|endoftext|> is the EOS token of ProtGPT2.

        <|endoftext|>
        ......(60 amino acids)......
        ....
        <|endoftext|>
        """
        lines = [sequence[i:i + 60] for i in range(0, len(sequence), 60)]
        formatted_sequence = '\n'.join(lines)
        formatted_sequence = '<|endoftext|>\n' + formatted_sequence + '\n<|endoftext|>'

        return formatted_sequence

    def retrieve_fragments(self, fragments: list[str], sequence: str) -> list[Phrase]:
        # Modify fragments to allow \n between characters
        fragment_patterns = [re.compile(''.join(f"{re.escape(c)}\n?" for c in fragment)) for fragment in fragments]

        result = []
        last_index = 0
        sequence_length = len(sequence)

        while last_index < sequence_length:
            match = None
            match_start = sequence_length

            for pattern in fragment_patterns:
                m = pattern.search(sequence, last_index)
                if m and m.start() < match_start:
                    match = m
                    match_start = m.start()

            if match:
                if match_start > last_index:
                    result.append(Phrase(content=sequence[last_index:match_start], is_phrase=False))

                phrase_type = None
                phrase_description = None
                for phrase in self.fragment_mappings[sequence]:
                    if phrase["phrase"] == match.group(0).replace("\n", ""):
                        phrase_type = phrase["type"]
                        phrase_description = phrase["description"]
                        break
                result.append(Phrase(
                    content=sequence[match_start:match.end()],
                    is_phrase=True,
                    type=phrase_type,
                    description=phrase_description,
                ))
                last_index = match.end()
            else:
                result.append(Phrase(content=sequence[last_index:], is_phrase=False))
                break

        return result

    def sample(self, document: Document) -> list[Phrase]:
        sequence = self.format_amino_acid_sequence(document.content) if self.format_sequence else document.content
        fragments = self.get_fragments(sequence)
        return self.retrieve_fragments(fragments, sequence)
