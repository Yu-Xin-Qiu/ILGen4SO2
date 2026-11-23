# ILGen4SO2

Code and data for property-conditioned generation of ionic liquids for selective SO2 capture.

The framework combines a SELFIES–Transformer VAE conditioned on melting point,
viscosity and toxicity with predictor feedback and multi-objective screening.
Feedback uses R = AC × DE × S / SA_max, combining absorption capacity, the
absorption-to-desorption solubility ratio, SO2/CO2 selectivity and synthetic
accessibility. Global and local sampling explore candidates around and beyond
the reference ionic liquid [BMIM][MeSO4].

Across three random seeds, final sampling produces 30,000 attempts and 10,285
distinct physicochemically feasible ionic liquids. Strict screening retains
15 candidates, of which 11 form the four-objective Pareto front.

## Data and code

- Root CSV files: experimental melting-point, viscosity, toxicity and gas-solubility data.
- `data/generated/`: feedback predictions, final samples and screening results.
- `data/processed/`: feasible candidates and the four highest-R Pareto candidates.
- `data/figure_tables.json.gz`: source tables for the figures.
- `data/quantum_chemistry.json`: optimized structures and Gaussian calculation inputs.
- `work/text_generation_study/`: model architectures, training, prediction and feedback code.
- `previous/`: property-predictor architectures and tokenizers.
- `configs/current_protocol.json`: physicochemical constraints and sampling parameters.

## Usage

Use Python 3.10 or newer and run from the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_results.py
python scripts/plot_results.py
```

Results and 600-dpi figures are saved to `reproduced/`.
Export figure tables as CSV files and quantum-chemistry inputs with:

```bash
python scripts/export_source_data.py
```

## Models

Physicochemical property predictors build on [ILBERT](https://github.com/Yu-Xin-Qiu/ILBERT),
with the original model and data archive available on
[Zenodo](https://doi.org/10.5281/zenodo.14601047).
Gas-solubility prediction models are from [IL4GAS](https://github.com/Yu-Xin-Qiu/IL4GAS).

See [model resources](docs/MODELS.md) for the models used in this study and
[methods and data](docs/METHODS_AND_DATA.md) for objective definitions and screening criteria.
