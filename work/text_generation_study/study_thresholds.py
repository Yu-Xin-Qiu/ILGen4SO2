"""Property thresholds for the updated full-fold ionic-liquid study."""
TM_LIMIT_K = 298.15
VISCOSITY_LIMIT_MPAS = 200.0
TOXICITY_MIN = 2.0

PHYSICAL_LIMITS = {
    "Tm_K_lt": TM_LIMIT_K,
    "viscosity_mPas_lt": VISCOSITY_LIMIT_MPAS,
    "log10_EC50_gt": TOXICITY_MIN,
}
