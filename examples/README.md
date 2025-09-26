# Details of the configuration
Different commands need different argument sets.
- chat: infer, model
- train: train, data, model
- eval: eval, infer, model 

## eval
- `test_data_file(str)`: The data path for evaluation.
- `batch_size(int)`: The batch size of evaluation for one step.
- `task_type(EvalTaskType)`: The evaluation task type.(more details in [eval_args.py](../src/dvagen/configs/eval_args.py) )
- `eval_seed(int)`: The ramdom seed for evaluation.
- `save_results_path(str)`: The path for saving results.

## data
- `train_path(str)`: The path of training data.
- `validation_path(str)`: The path of validation data.
- `max_text_length(int)`: The max sequence length for the text encoder.
- `max_sequence_length(int)`: The max protein sequence length.
- `max_phrase_length(int)`: The max protein fragment length.

## model
- `phrase_sampler_type(PhraseSamplerType)`: The type of phrase sampler including `FMM`, `N_TOKENS`, `N_WORDS`.(more details in [model_args.py](../src/dvagen/configs/model_args.py))
- `sampler_model_path(str)`: The model path for phrase sampler to tokenize text.
- `sampler_random_up(int)`: The max length of phrase gap.
- `sampler_random_low(int)`: The min length of phrase gap.
- `phrase_max_length(int)`: The max length of phrase.

## infer
- `doc_top_k(int)`: The top K supporting documents to retrieve for each query.
- `embedding_model_path(str)`: The path for the embedding model used in retrieval.
- `data_file(str)`: The data path of the supporting documents.
- `vector_store_path(str)`: The path of the vector store index.
- `save_vector_store_path(str)`: The save path of the vector store index.
- `protein_sequence_mapping_file(str)`: The path of a json file that maps textual descriptions to protein sequences. See below for more details.

The same setting as `generate` method in huggingface.
- `do_sample(bool)`: Whether or not to use sampling ; use greedy decoding otherwise.
- `temperature(float)`: The value used to module the next token probabilities. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0
- `max_length(int)`: The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
- `max_new_tokens(int)`: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- `top_k(int)`: The number of highest probability vocabulary tokens to keep for top-k-filtering. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 50.
- `top_p(float)`:If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0

## train

- Includes same arguments as huggingface TrainingArguments.
- `text_encoder_path(str)`: The path for the text encoder.
- `language_model_path(str)`: The path for the protein language model backbone.
- `phrase_encoder_path(str)`: The path for the fragment encoder.
- `freeze_text_encoder(bool)`: Whether to freeze the text encoder during training.
- `freeze_language_model(bool)`: Whether to freeze the protein language model during training.
- `protein_fragment_mapping_file(str)`: The path of a json file that maps protein sequences to their corresponding fragments (with functional annotations). See below for more details.
- `use_text_encoder_proj(bool)`: Whether to use a MLP projection layer after the text encoder.
- `use_phrase_encoder_proj(bool)`: Whether to use a MLP projection layer after the fragment encoder
- `use_type_loss(bool)`: Whether to use $\mathcal{L}_{text{TYPE}}$ (for more details, please check Section 3.2 in our paper). The weight assigned to each type is automatically added.
- `type_loss_weight(bool)`: The weight for the type loss. ($\alpha$ in Equation 6)
- `use_description_loss(bool)`: Whether to use $\mathcal{L}_{text{DESC}}$ (for more details, please check Section 3.2 in our paper).
- `description_loss_weight(float)`: The weight for the description loss. ($\beta$ in Equation 6)
- `finetuning_type(str)`: Training method for model including "freeze"(freeze backbone model), "lora", "full"(full finetune).
- `r(int)`: Lora attention dimension (the “rank”).
- `alpha(int)`: The alpha parameter for Lora scaling.
- `dropout(float)`: The dropout probability for Lora layers.
- `target_modules(list)`: The names of the modules to apply the adapter to. If this is specified, only the modules with the specified names will be replaced. When passing a string, a regex match will be performed. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as ‘all-linear’, then all linear/Conv1D modules are chosen (if the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules(the same as LoraConfig in huggingface)

## Examples of the files

We use two json files to map:
1. Protein descriptions to sequences.
2. Protein sequences to fragments (with functional annotations).

For the first mapping file (also used as the training file), the format is as follows:
```json
{
  "instruction": "Plays a role in virus cell tropism, [...]",
  "sequence": "MVRLFYNPIKY [...]"
}
```

For the second mapping file, the format is as follows:
```json
{
  "sequence": "MKNCEYQQIDPRALRTPSSR [...]",
  "phrases": [
    {
      "phrase": "KLKYCFTCKM [...]",
      "type": "DOMAIN",
      "name": "Palmitoyltrfase_DHHC",
      "description": "Palmitoyltransferase, DHHC domain"
    }
  ]
}
```