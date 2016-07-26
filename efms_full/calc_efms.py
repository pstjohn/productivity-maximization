from Models.succinogenes import v6_new_biomass_atpm
from Models.ecoli import ecoli_core_biomass

succ = v6_new_biomass_atpm()
ecoli = ecoli_core_biomass()

from cobra.elementary_flux_modes import calculate_elementary_modes

succ_efms = calculate_elementary_modes(succ, verbose=False)
succ_efms = succ_efms.divide(-succ_efms.EX_glc_e, 0)
succ_efms.dropna().to_pickle('succ_efms.p')

ecoli_efms = calculate_elementary_modes(ecoli, verbose=False)
ecoli_efms = ecoli_efms.divide(-ecoli_efms.EX_glc_e, 0)
ecoli_efms.dropna().to_pickle('ecoli_efms.p')
