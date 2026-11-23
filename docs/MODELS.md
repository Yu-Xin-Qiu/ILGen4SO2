# Models

## ILBERT

- [Source code](https://github.com/Yu-Xin-Qiu/ILBERT)
- [Model and data archive](https://doi.org/10.5281/zenodo.14601047)
- [Associated publication](https://doi.org/10.1039/D5DD00035A)

The physicochemical property predictors use ILBERT architectures and tokenizers. Their definitions
and configurations are provided in `previous/`.

## IL4GAS

- [Source code and gas-solubility models](https://github.com/Yu-Xin-Qiu/IL4GAS)

The SO2 and CO2 solubility prediction models are from IL4GAS. The gas-predictor
architecture and tokenizer used by the screening code are provided in `previous/IL4SO2/`.

## Models used in this study

The prediction ensemble comprises ten folds for each physicochemical property
and five folds for each gas. The study also uses an initial conditional generator
and three selected round-4 generators. Checkpoint paths and SHA-256 hashes are
listed in `model_weights_manifest.csv`.

The exact checkpoints used for this study are identified in the manifest.
The updated predictors and conditional generators use study-specific training.
Contact the authors for access to these checkpoints.

## Running model feedback

Install `requirements-training.txt` and place the checkpoints at the paths
specified in `model_weights_manifest.csv`. Model feedback requires CUDA.

```bash
python scripts/run_feedback.py --rounds 9 --seed 137
```

Omit `--seed` to run all three seeds. Saved-result analysis uses the released
prediction tables and can be run with `scripts/reproduce_results.py`.
