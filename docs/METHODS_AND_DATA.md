# Methods and populations

Point-prediction hard constraints: Tm < 298.15 K, viscosity at 298.15 K < 200 mPa s,
IPC-81 log10 EC50 > 2. Ten folds per physical predictor and five folds per gas.
Gas predictions are averaged on the ln(x) scale before exponentiation.
AC = xSO2,abs; DE = xSO2,abs/xSO2,des; S = xSO2,abs/xCO2,abs.
Absorption is 298.15 K, desorption 313.15 K, pressure 1 bar.
SA_max = max(cation SA, anion SA); R = AC*DE*S/SA_max.

R feedback requires feasible physical properties, physically valid gas predictions,
absence from the experimental library and R above [BMIM][MeSO4]. Final strict
screening instead additionally requires each AC/DE/S above the reference and
SA_max below it. Pareto maximizes AC/DE/S and minimizes SA_max; ranking is descending R.
Folds' SD and historical uncertainty utilities can remain in shared code, but no
uncertainty gate is used by the paper-aligned analysis.

Three seeds 137/251/367; per seed and round 5,000 global plus 5,000 local attempts;
local sigma 0.25, temperature 1.0 and top_k 32. The paper analyzes batches 0-9.
Round 0 is the initial model; round-4 generation uses round_03/model.pt, which
was saved AFTER the update from batch 3. Updates use two epochs at learning rate 3e-5.
Saved fresh samples total 30,000 attempts, 10,285 distinct property-feasible ILs,
15 strict candidates and 11 Pareto members. TOP4: IL0006, IL0014, IL0003, IL0008.

Raw experimental CSVs preserve observations, repeated measurements and original
state conditions. Experimental viscosity/gas distributions pool available states;
generated values refer to the stated temperature/pressure. Identity uses pair InChIKey.
Raw viscosity pair identities use `IL SMILES`, not `cation SMILES`.

Generated property tables report ensemble point estimates. Figure source tables
and quantum-chemistry inputs can be exported with scripts/export_source_data.py.
Figure export numbers predate manuscript renumbering: export Figure06 = current
feedback Figure5, export Figure07 = sampling Figure6, feasible Figure08 = current
population Figure7, export Figure09 = current Pareto Figure8.

