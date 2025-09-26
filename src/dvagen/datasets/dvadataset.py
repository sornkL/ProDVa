import json
import logging

import datasets
from torch.utils.data import Dataset
from tqdm import tqdm

from ..models.tokenization_dva import DVATokenizer


logger = logging.getLogger(__name__)


class DVADataset(Dataset):
    def __init__(
        self,
        tokenizer: DVATokenizer,
        protein_fragment_mapping_file: str = None,
        data_path: str = None,
        save_data_path: str = None,
        max_sequence_length: int = 512,
        max_phrase_length: int = 512,
        cut_len: int = -1,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.protein_fragment_mapping_file = protein_fragment_mapping_file
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.max_phrase_length = max_phrase_length

        if self.protein_fragment_mapping_file is not None:
            with open(self.protein_fragment_mapping_file) as f:
                self.fragment_mappings = json.load(f)

            self.fragment_mappings = {item["sequence"]: item["phrases"] for item in self.fragment_mappings}
            self.types = set()
            for sequence, phrases in self.fragment_mappings.items():
                for phrase in phrases:
                    self.types.add(phrase['type'])
            self.phrase_types = list(self.types)
            self.phrase_types.sort()  # Make the types ordered and deterministic
            self.phrase_types = {phrase_type: i for i, phrase_type in enumerate(self.phrase_types)}

        if self.data_path is not None and self.data_path != "None":
            # Load the dataset from the saved path
            self.dataset = self.data_process()
            if save_data_path is not None:
                self.data_save(self.dataset, save_path=save_data_path)
        else:
            self.dataset = self.data_load(save_path=save_data_path)
        self.dataset = self.dataset.select(range(cut_len)) if cut_len > 0 else self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        instruction = self.dataset[idx]["instruction"]
        sequence = self.dataset[idx]["sequence"]
        inputs = self.tokenizer.tokenize(
            sequence, max_sequence_length=self.max_sequence_length, max_phrase_length=self.max_phrase_length
        )
        for phrase in inputs:
            if phrase.is_phrase:
                phrase.type = self.phrase_types[phrase.type]
        return instruction, inputs

    def data_process(self) -> datasets.Dataset:
        dataset = []
        with open(self.data_path) as f:
            data = json.load(f)

        for item in data:
            instruction = item["instruction"].strip()
            sequence = item["sequence"].strip()
            dataset.append({
                "instruction": instruction,
                "sequence": sequence,
            })

        return datasets.Dataset.from_list(dataset)

    def get_type_weight(self, verbose: bool = False):
        type_counts = [0 for _ in range(len(self.phrase_types))]
        for i in tqdm(range(len(self)), disable=not verbose, desc="Computing the type weight."):
            inputs = self[i][1]  # get the inputs of dataset[i]
            for phrase in inputs:
                if phrase.is_phrase:
                    type_counts[phrase.type] += 1

        type_weights = [1.0 / count if count > 0 else 0.0 for count in type_counts]
        type_weights = [weight / sum(type_weights) for weight in type_weights]
        return type_weights

    @staticmethod
    def data_save(dataset: datasets.Dataset, save_path: str):
        """
        Save the processed data to a specified path.
        This function is a placeholder and should be implemented based on specific requirements.
        """
        # Example implementation
        dataset.to_json(save_path, orient="records", lines=True)
        return

    @staticmethod
    def data_load(save_path: str):
        return datasets.load_dataset("json", data_files=save_path, split="train")
