# Data dictionary

Original files are retained in the repository root. Derived files are not experiments.

| File | Value column | Meaning / units |
| --- | --- | --- |
| melting point.csv | MP_K | Melting point, kelvin |
| viscosity.csv | n_mPas / ln(n_mPas) | Viscosity, mPa s / natural log viscosity |
| toxicity.csv | logEC50 | IPC-81 log10 EC50, retaining original assay concentration convention |
| SO2 solubility.csv | ln(x_SO2) | Natural log of SO2 mole fraction |
| CO2 solubility.csv | ln(x_CO2) | Natural log of CO2 mole fraction |

`T/K` is temperature in kelvin and `P/bar` is pressure in bar. Raw viscosity
includes bibliographic/dataset identifiers, acceptance flags and alternative
SMILES columns. These are preserved, not silently filtered. `IL SMILES` is used
for raw viscosity pair counts, not `cation SMILES`.

`processed/experimental_properties_long.csv` provides one row per original
observation, a source-file/source-row link, original value and original column,
the physical value (`exp(ln(x))` for gases), units and recorded conditions.
Empty temperature/pressure fields mean unavailable or inapplicable, not zero.
The experimental rows are not deduplicated across conditions.

Generated records carry `cation_smiles`, `anion_smiles`, `IL_SMILES`, `InChIKey`,
`Tm_K`, `eta_298_mPas`, `toxicity` (log10 EC50), `SA_cation`, `SA_anion`,
`SA_max`, `AC`, `DE`, `S`, flags and fold-level `ln_x_*_foldNN` values.
`library_novel` means absent from the original experimental pair library.
`hard_pass` uses the three physicochemical point thresholds. `joint_hit` is
the final strict multiobjective screen, NOT feedback R eligibility.
Full CSV header inventory is in `column_index.csv`.
