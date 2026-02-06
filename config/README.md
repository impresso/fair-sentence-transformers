# Config folder overview

This directory provides sample configuration files for the main steps. Use them as templates and adjust paths and parameters to your environment.

## Layout

- `tokenization_config_wiki_parallel.json`
	- Example config for tokenizing the multilingual Wikipedia dataset on disk using `src/fair_sentence_transformers/scripts/tokenize_dataset.py`.

- `wiki_parallel/`
	- `indices_wiki_parallel_1_en_de_hi_it_ko_zh.config.json`: Example generator config for creating parallel indices with `src/fair_sentence_transformers/scripts/create_parallel_indices.py`.
	- `indices_wiki_parallel_1_en_de_hi_it_ko_zh.json`: Example generated indices file (concat and standalone indices + generation metadata).

- `Alibaba_mGTE/`
	- `wiki_parallel/`: Example embedding configs for running Experiments 1 and 2 with `src/fair_sentence_transformers/scripts/compute_embeddings.py` using the “wiki_parallel” mode. Files are organized by experiment index and language setting (monolingual and mixed-language documents).

## Tokenization configs

Use `tokenization_config_wiki_parallel.json` with:

```bash
poetry run python src/fair_sentence_transformers/scripts/tokenize_dataset.py --config config/tokenization_config_wiki_parallel.json
```

## Parallel indices (experiment setup)

Create indices with the generator config under `wiki_parallel/`:

```bash
poetry run python src/fair_sentence_transformers/scripts/create_parallel_indices.py --config config/wiki_parallel/indices_wiki_parallel_1_en_de_hi_it_ko_zh.config.json
```

This produces a single JSON (also under `wiki_parallel/`) containing:
- `concat_indices`: permutations of segment indices to concatenate
- `standalone_indices`: the set of used segment indices
- `generation_config` and summary stats

Notes:
- Adjust language lists, `concat_size`, length constraints, and `sample_size` to your experiment.
- Keep indices fixed across models for comparability.

## Embedding experiments (Exp1 & Exp2)

The model-specific folder (e.g., `Alibaba_mGTE/wiki_parallel/`) contains example configs for `src/fair_sentence_transformers/scripts/compute_embeddings.py`.

Run with:

```bash
poetry run python src/fair_sentence_transformers/scripts/compute_embeddings.py --config <path-to-config.json>
```

Typical fields:
- `model_name`, `tokenized_dataset_path`, `indices_path`, `mode` ("wiki_parallel")
- `source_lang` and (optionally) `target_lang` for mixed-language documents
- `batch_size_standalone`, `batch_size_concat`, `device`, `separator`

Mixed-language convention:
- `target_lang` is the language of the first segment in each concatenated document.
- `source_lang` is the language of all remaining segments (2...n).

### Applying Attention Calibration

Embedding configs may include calibration keys to enable attention calibration during embedding computation (applies to Exp1/2):
- `apply_attn_calibration`, `calib_layers`, `calib_source_tokens`, `calib_basket_size`
- Per-embedder overrides are also supported (see main README for details).

## See also

- Top-level `README.md` for the full workflow, experiments, analysis, and visualization.
- `src/fair_sentence_transformers/scripts/create_wiki_parallel_configs.sh` to auto-generate many embedding configs; `run_all_configs.sh` to execute a folder of configs.
