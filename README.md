# ProDVa

This is the official implementation of the paper: [*Protein Design with Dynamic Protein Vocabulary*](https://arxiv.org/pdf/2505.18966).

We refactored the codebase using the [DVAGen](https://github.com/AntNLP/DVAGen) framework (🎉recently accepted to EMNLP 2025 Demo).

# Updates

- [2025/09/19] 🎉🎉ProDVa was accepted to NeurIPS 2025 Spotlight!

# Quick Start

## Setup

Download the repository and install ProDVa:

```bash
git clone https://github.com/sornkL/ProDVa.git
cd DVAGen
pip install -e .
```

Download the model weights from [🤗HuggingFace ProDVa Collection](https://huggingface.co/collections/nwliu/prodva-68d64df8e1e7ebc88f314692).
All datasets and model weights are provided in the Collection, with details listed in the tables below.

Datasets used in our paper (see Appendix C.2 for more details):

| Datasets              | Task Descriptions                                      | Links                                                                   |
|-----------------------|--------------------------------------------------------|-------------------------------------------------------------------------|
| CAMEO                 | Protein Design from Function Keywords (Section 4.2)    | [Download](https://huggingface.co/datasets/nwliu/CAMEO)                 |
| Molinst-SwissProtCLAP | Protein Design from Textual Descriptions (Section 4.3) | [Download](https://huggingface.co/datasets/nwliu/Molinst-SwissProtCLAP) |

Each dataset includes training/validation/test sets, a fragment mapping file, and a FAISS index directory constructed from the training set.

Model weights used in our paper:

| Models                       | Descriptions                                          | Links                                                                 |
|------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------|
| ProDVa-CAMEO                 | ProDVa trained on CAMEO (Section 4.2)                 | [Download](https://huggingface.co/nwliu/ProDVa-CAMEO)                 |
| ProDVa-Molinst-SwissProtCLAP | ProDVa trained on Molinst-SwissProtCLAP (Section 4.3) | [Download](https://huggingface.co/nwliu/ProDVa-Molinst-SwissProtCLAP) |

## Designing Proteins with ProDVa

Simply use the following command to launch a CLI chat tool for designing proteins.

```bash
dvagen chat --config_path examples/chat.yaml
```

## Evaluation

The evaluation pipeline typically involves two stages: protein design and assessment using certain metrics.
Run the following command to launch the pipeline:

```bash
dvagen eval --config_path examples/eval.yaml
```

This command will prompt ProDVa to design proteins using the given test set.
Note that for evaluation (or benchmarking the results), please refer to our latest work: [PDFBench](https://github.com/PDFBench/PDFBench).
We are working on integrating the evaluation metrics into our codebase.

## Training

By default, we use `deepspeed` to launch the training script. To train ProDVa, use the following command:

```bash
dvagen train [deepspeed_args] --config_path examples/train.yaml
```

Details of the configuration files are available in the [examples/README.md](examples/README.md) file.
We also include some file (e.g., training set, mapping file) examples in the README.
Most of the arguments are directly inherited from DVAGen while we ignore the unused ones.
Some newly added arguments  specific to ProDVa are also documented.

# Citation

```bibtex
@article{liu2025protein,
  title={Protein Design with Dynamic Protein Vocabulary},
  author={Liu, Nuowei and Kuang, Jiahao and Liu, Yanting and Sun, Changzhi and Ji, Tao and Wu, Yuanbin and Lan, Man},
  journal={arXiv preprint arXiv:2505.18966},
  year={2025}
}
```

If you find the codebase helpful, please also cite the DVAGen framework:

```
Coming Soon!
```