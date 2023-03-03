import numpy as np
import pandas as pd
from expt2 import expt2


cols = pd.MultiIndex.from_product([['NSW_mean', 'NSW_std', 'NAU_mean', 'NAU_std', 'NAU_ND_mean', 'NAU_ND_std'], list(range(1,6))], names=['measure', 'heuristic'])
df = pd.DataFrame(columns=cols)

num_agents = 100
num_heuristics = 5
ix = [1]
ix.extend(np.arange(0.1,0.7,0.1)*num_agents)
MW = np.zeros((len(ix), num_heuristics))
SW = np.zeros((len(ix), num_heuristics))
MU = np.zeros((len(ix), num_heuristics))
SU = np.zeros((len(ix), num_heuristics))
MU_ND = np.zeros((len(ix), num_heuristics))
SU_ND = np.zeros((len(ix), num_heuristics))
for itr in range(len(ix)):
	alpha = ix[itr]
	overall_opt, welfare, utility, utility_nd = expt2(int(alpha))
	MW[itr,:] = np.mean(welfare[1:,:]/np.expand_dims(overall_opt[1:], -1), 0)
	SW[itr,:] = np.std(welfare[1:,:]/np.expand_dims(overall_opt[1:], -1), 0)
	MU[itr,:] = np.mean(utility[1:,:],0)
	SU[itr,:] = np.std(utility[1:,:],0)
	MU_ND[itr,:] = np.mean(utility_nd[1:,:],0)
	SU_ND[itr,:] = np.std(utility_nd[1:,:],0)


	df[cols[:5]] = MW
	df[cols[5:10]] = SW
	df[cols[10:15]] = MU
	df[cols[15:20]] = SU
	df[cols[20:25]] = MU_ND
	df[cols[25:30]] = SU_ND

	df.to_csv('data.csv')
