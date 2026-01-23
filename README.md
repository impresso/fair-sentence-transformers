# [Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias](link_not_yet_existent) - FairSentenceTransformers and Replication ![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg) 

---

## Overview

This repository accompanies our [preprint(under-review)](link_not_yet_existent) source code for the examination of positional bias in long documents of multilingual embedding models.

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

Within our work, we introduce an inference-time attention calibration we wrap this as an extension of Sentence Transformers, named Fair Sentence Transformers. Fair Sentence Transformers goal is to:

1. Include different inference-time calibration techniques aimed to maximize fairness within embedding models.
2. Enable support of existing and future embedding models by creating "Generic" implementations that can be configured to each specific model's attributes.

### Example use:


### Supported Models and Methods

encode_positionally_fair - Inference time attention calibration to represent input from all positions

Tested Models:

[Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

Qwen3-Embedding Family: [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

Supported - Our current implementations can very easily support more models, just requires configuration and testing.

---

## Reproducing the Experiments

To fully replicate our results, please follow the instructions decipited in our Replication readme (REPL_README.md)

## Citation

If you use these resources, please cite our paper:

```bibtex
ARXIV bibtex TO BE ADDED
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
