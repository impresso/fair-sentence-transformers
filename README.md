# [Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias](https://arxiv.org/abs/2601.16934) - FairSentenceTransformers and Replication ![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg) 

---

## Overview

This repository accompanies our [preprint(under review)](https://arxiv.org/abs/2601.16934) with source code for the examination of positional bias in long documents of multilingual embedding models and the fair-sentence-transformers extension.

---

## Table of Contents

- [Overview](#overview)
- [Fair Sentence Transformers](#fair-sentence-transformers)
- [Repository Structure](#repository-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Citation](#citation)
- [About Impresso](#about-impresso)
- [License](#license)

---

## Fair Sentence Transformers

We introduce an inference-time attention calibration method, implemented as an extension of Sentence Transformers called Fair Sentence Transformers. This tool aims to:

1. Provide a Wrapper Class for inference-time calibration techniques that improve fairness in embedding models.
2. Support existing and future embedding model releases through generic implementations configurable to each model's attributes.

### Setup and Example use:

```bash
poetry install
```

```python
from src.locobench.core.fair_sentence_transformer import FairSentenceTransformer
 
input_texts = [
    "What is the capital of Switzerland?",
    "How to make an omelette?",
    "Wie viele Einwohner hat Deutschland?",
]
 
model_name_or_path = "Alibaba-NLP/gte-multilingual-base"
model = FairSentenceTransformer(model_name_or_path)
 
# Standard SentenceTransformer embeddings
embeddings = model.encode(input_texts)  # shape: (3, 768)
 
# Fair SentenceTransformer embeddings
fair_embeddings = model.encode_positionally_fair(
    input_texts,
    calib_strength=0.5,
    calib_basket_size=128,
    calib_layers=6,
)  # shape: (3, 768)
```
pip install coming soon

### Supported Models and Methods

**encode_positionally_fair** - Inference-time attention calibration to ensure fair representation of input from all positions.

**Tested Models:**
- [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- Qwen3-Embedding Family: [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

**Extensibility:** Our implementation can support additional models with a configuration and quick test of new additions. Feel free put a pull request with your favourite model.

---

## Reproducing the Experiments

To fully replicate our results, please follow the instructions decipited in our Replication readme (REPL_README.md)

## Citation

If you use these resources, please cite our paper:

```bibtex
@misc{schuhmacher2026informationrepresentationfairnesslongdocument,
      title={Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias}, 
      author={Elias Schuhmacher and Andrianos Michail and Juri Opitz and Rico Sennrich and Simon Clematide},
      year={2026},
      eprint={2601.16934},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.16934}, 
}
```

## About Impresso

### Impresso project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017-2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

### Copyright

Copyright (C) 2026 The Impresso team.

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>
