import torch
from transformers import AutoTokenizer

from .phrase import Document, Phrase


class DVATokenizer:
    def __init__(
        self,
        static_vocab: int,
        text_encoder_name_or_path: str,
        model_name_or_path: str,
        phrase_encoder_name_or_path: str,
        sampler,
        **kwargs,
    ):
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name_or_path)
        self.lm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.phrase_tokenizer = AutoTokenizer.from_pretrained(phrase_encoder_name_or_path)
        self.static_vocab = static_vocab
        self.sampler = sampler

        if self.text_tokenizer.pad_token_id is None:
            self.text_tokenizer.pad_token_id = self.text_tokenizer.eos_token_id
        if self.lm_tokenizer.pad_token_id is None:
            self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id
        if self.phrase_tokenizer.pad_token_id is None:
            self.phrase_tokenizer.pad_token_id = self.phrase_tokenizer.eos_token_id

    def tokenize(self, text: str, max_sequence_length: int = None, max_phrase_length: int = None) -> list[Phrase]:
        if max_sequence_length is not None:
            text_ids = self.lm_tokenizer(text, truncation=True, max_length=max_sequence_length)["input_ids"]
            text = self.lm_tokenizer.decode(text_ids, skip_special_tokens=True)
        doc = Document(content=text)
        phrases = self.sampler.sample(doc)
        if max_phrase_length is not None:
            for phrase in phrases:
                if phrase.is_phrase:
                    phrase_ids = self.phrase_tokenizer(phrase.content, truncation=True, max_length=max_phrase_length)["input_ids"]
                    phrase.content = self.phrase_tokenizer.decode(phrase_ids, skip_special_tokens=True)
        return phrases

    #
    def encode(self, phrases: list[Phrase], **kwargs):
        input_ids = []
        phrase_ids = []  # list[list[int]]
        for phrase in phrases:
            if phrase.is_phrase:
                input_ids.append(self.static_vocab + len(phrase_ids))  # Unique ID for phrase
                phrase_ids.append(self.phrase_tokenizer.encode(phrase.content, add_special_tokens=False))
            else:
                input_ids.extend(self.lm_tokenizer.encode(phrase.content, add_special_tokens=False))
        return {
            "input_ids": input_ids,
            "phrase_ids": phrase_ids,
        }

    def batch_encode(self, phrases_list: list[list[Phrase]], phrases_mask: bool, **kwargs):
        input_ids = []
        phrase_ids = []  # list[list[int]]
        for phrases in phrases_list:
            output = self.encode(phrases)
            _input_ids, _phrase_ids = output["input_ids"], output["phrase_ids"]
            input_ids.append(_input_ids)
            phrase_ids.append(_phrase_ids)
        padded_input_ids = self.lm_tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = padded_input_ids["input_ids"]
        attention_mask = padded_input_ids["attention_mask"]
        sum = 0
        for i in range(len(phrases_list)):
            input_ids[i] = torch.where((input_ids[i] >= self.static_vocab), input_ids[i] + sum, input_ids[i])
            sum += len(phrase_ids[i])
        combined_phrase_ids = []
        for i in range(len(phrase_ids)):
            combined_phrase_ids.extend(phrase_ids[i])
        padded_phrase_ids = self.phrase_tokenizer.pad(
            {"input_ids": combined_phrase_ids},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        outputs = {
            "input_ids": input_ids,
            "phrase_ids": padded_phrase_ids["input_ids"],
            "attention_mask": attention_mask,
            "phrase_attention_mask": padded_phrase_ids["attention_mask"],
        }
        if phrases_mask:
            outputs["mask_ids"] = []
            mask = list(range(self.static_vocab, self.static_vocab + sum))
            now = 0
            for _phrase_ids in phrase_ids:
                outputs["mask_ids"].append(mask[:now] + mask[now + len(_phrase_ids) :])
                now += len(_phrase_ids)
        return outputs

    def decode(
        self, input_ids: list[int], phrases_ids: list[list[int]] = None, return_ids: bool = False, **kwargs
    ) -> dict:
        token_ids = []
        token_phrase_ids = []
        for id in input_ids:
            if id < self.static_vocab:
                token_ids.append(id)
                token_phrase_ids.append(id)
            else:
                phrase_index = id - self.static_vocab
                token_phrase_ids.append(
                    [
                        token_id
                        for token_id in phrases_ids[phrase_index]
                        if token_id != self.phrase_tokenizer.pad_token_id
                    ]
                )
                assert phrase_index < len(phrases_ids), "Invalid phrase index"
                phrase = self.phrase_tokenizer.decode(phrases_ids[phrase_index], skip_special_tokens=True)
                token_ids.extend(self.lm_tokenizer.encode(phrase, add_special_tokens=False))

        decoded_sentence = self.lm_tokenizer.decode(token_ids, skip_special_tokens=True)

        return {
            "decoded_sentence": decoded_sentence.replace("\n", ""),
            "ids": token_phrase_ids if return_ids else None,
        }

    def save_pretrained(self, save_directory: str):
        self.text_tokenizer.save_pretrained(f"{save_directory}/text_tokenizer")
        self.lm_tokenizer.save_pretrained(f"{save_directory}/lm_tokenizer")
        self.phrase_tokenizer.save_pretrained(f"{save_directory}/phrase_tokenizer")

    def update_dv(self, dv: dict[str, list[int]]):
        pass
