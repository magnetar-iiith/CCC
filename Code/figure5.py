import numpy as np
from itertools import chain, combinations

def expt2(alpha):

	def powerset(iterable):
		"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		s = list(iterable)
		return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

	num_agents = 100
	num_proj = 10

	num_heuristics = 5
	num_instances = 5000
	num_INST = 0
	welfare = np.zeros((num_instances+1, num_heuristics))
	utility = np.zeros((num_instances+1, num_heuristics))
	utility_nd = np.zeros((num_instances+1, num_heuristics))
	overall_opt = np.zeros((num_instances+1))

	def generateGame():
		flag = 0

		## UNIFORM
		theta = np.random.uniform(0, 10, size=(num_agents,num_proj))

		## EXPONENTIAL
		# theta = np.random.exponential(scale = 1.5, size=(num_agents,num_proj))+5

		# NORMAL
		# theta = np.zeros((num_agents, num_proj))
		# means = np.random.uniform(1, 10, size = (num_proj))
		# var = np.random.uniform(1, 10, size=(num_proj))
		# for j in range(num_proj):
		# 	theta[:,j] = np.random.normal(loc = means[j], scale = var[j], size=(num_agents))
		# theta = theta - np.min(theta)

		## Identical 
		# theta = np.random.uniform(0, 10, size=(1,num_proj))
		# theta = np.tile(theta, (num_agents, 1))
		# print(theta.shape)

		vartheta = np.sum(theta, 0)
		targets  = np.random.uniform(0.5, 1, size=(num_proj)) * vartheta
		bonuses = 0.75 * (vartheta - targets)
		
		budget_val = np.random.uniform(0.4, 0.8) * np.sum(targets)
		# optimal welfare
		max_welfare = 0
		max_S = []
		for S in powerset(range(num_proj)):
			S = list(S)
			if (sum(targets[S]) <=  budget_val):
				wel_ = np.sum(np.sum(theta[:,S],0) - targets[S])
				if(wel_ >= max_welfare):
					max_welfare = wel_
					max_S = S
		### PPR contributions
		x = targets*theta/(targets+bonuses)

		### budgetsx
		budgets = np.sum(x[:,max_S], 1) 
		if (budget_val - np.sum(budgets) >= 0):
			prob = np.random.uniform(size=num_agents)
			prob = prob/np.sum(prob)
			budgets += prob*(budget_val - np.sum(budgets))
		else:
			flag = 1
		return [theta, targets, bonuses, budgets, x, max_S, max_welfare, flag]

	while(True):
		[theta, targets, bonuses, budgets, x_ppr, max_S, max_welfare, flag] = generateGame()
		if(flag):
			continue
		# print("****** Theta *******")
		# print(theta)
		# print("****** Target *******")
		# print(targets)
		# print("****** budget *******")
		# print(budgets)
		# print("****** bonus *******")
		# print(bonuses)
		num_INST += 1
		if (num_INST > num_instances):
			break
		
		contributions_per_agent = np.zeros((num_heuristics, num_agents, num_proj))
		contributions = np.zeros((num_heuristics, num_proj))

		## calculate optimal social welfare
		overall_opt[num_INST] = max_welfare

		############################################## zeroth agent deviates ######################################################################
		###########################################################################################################################################
		for i in range(alpha):
			### Heuristic 1: equally across projects
			cont_ = (contributions[0,:]<targets) * np.minimum(np.minimum(np.array([budgets[i]/num_proj]*num_proj), theta[i,:]), targets -  contributions[0,:])
			contributions_per_agent[0,i,:] = cont_
			contributions[0,:] +=  cont_

			### Heuristic 2: proportional to valuations
			cont_ = (contributions[1,:]<targets) * np.minimum(budgets[i]*theta[i,:]/sum(theta[i,:]), targets -  contributions[1,:])
			contributions[1,:] +=  cont_
			contributions_per_agent[1,i,:] =  cont_

			### Heuristic 3: ppr according to highest valuation
			ix = np.argsort(theta[i,:])[::-1]
			val = 0
			for j in ix:
				cont_ = (contributions[2, j]<targets[j]) * np.minimum(x_ppr[i, j], targets[j] - contributions[2, j])	 
				val += cont_	
				if(val <= budgets[i]):
					contributions_per_agent[2, i, j] = cont_
					contributions[2, j] += cont_
				else:
					val = val - cont_
					contributions_per_agent[2, i, j] = budgets[i] - val
					contributions[2, j] += budgets[i] - val
					break

			### Heuristic 4: ppr according to vartheta/target
			ix = np.argsort((np.sum(theta,0))/targets)[::-1]
			val = 0
			for j in ix:
				cont_ = (contributions[3, j]<targets[j]) * np.minimum(x_ppr[i, j], targets[j] - contributions[3, j])
				val += cont_
				if(val <= budgets[i]):
					contributions_per_agent[3, i, j] = cont_
					contributions[3, j] += cont_
				else:
					val = val - cont_
					contributions_per_agent[3, i, j] = budgets[i] - val
					contributions[3, j] += budgets[i] - val
					break

			### Heuristic 5: ppr according to max_S
			ix = max_S + [p for p in range(num_proj) if p not in max_S]
			val  = 0
			for j in max_S:
				cont_ = (contributions[4, j]<targets[j]) * np.minimum(x_ppr[i, j], targets[j] - contributions[4, j])
				val += cont_
				contributions_per_agent[4, i, j] = cont_
				contributions[4, j] += cont_

			ix = [p for p in range(num_proj) if p not in max_S]
			if (len(ix)>0):
				contributions_per_agent[4, i, ix] = (budgets[i] - val)/len(ix)
				contributions[4, ix] += (budgets[i] - val)/len(ix)

		###########################################################################################################################################
		###########################################################################################################################################
		
		for i in range(alpha, num_agents):
			### Every other agent contribute according to PPR in all heuristics
			for h in range(num_heuristics):
				val  = 0
				for j in max_S:
					cont_ = (contributions[h, j]<targets[j]) * np.minimum(x_ppr[i, j], targets[j] - contributions[h, j])
					contributions_per_agent[h, i, j] = cont_
					contributions[h, j] += cont_
					val += cont_
				ix = [p for p in range(num_proj) if p not in max_S]
				if len(ix) > 0:
					contributions_per_agent[h, i, ix] = (budgets[i] - val)/len(ix)
					contributions[h, ix] += (budgets[i] - val)/len(ix)

		final = (contributions/targets>=1)	
		### calculate overall welfare obtained
		for h in range(num_heuristics):
			welfare[num_INST, h] = sum(np.sum(final[h,:]*theta, 0) - targets*final[h,:])


		### Calculate agent utility for agents who deviated
		for h in range(num_heuristics):
			p =contributions_per_agent[h,:alpha,:]
			ix = np.where(contributions[h,:] != 0)[0]
			util = np.sum(final[h,:] * (theta[:alpha,:] - p), 1)
			util_ppr = np.sum(theta[:alpha, :] - x_ppr[:alpha, :], 1)
			if len(ix) > 0:
				util += np.sum((1-final[h,ix]) * p[:, ix] * bonuses[ix]/contributions[h,ix], 1)
				utility[num_INST, h] = np.mean(util/util_ppr)
		
		### Calculate agent utility for agents who did not deviate
		for h in range(num_heuristics):
			p =contributions_per_agent[h,alpha:,:]
			ix = np.where(contributions[h,:] != 0)[0]
			util = np.sum(final[h,:] * (theta[alpha:,:] - p), 1)
			util_ppr = np.sum(theta[alpha:, :] - x_ppr[alpha:, :], 1)
			if len(ix) > 0:
				util += np.sum((1-final[h,ix]) * p[:, ix] * bonuses[ix]/contributions[h,ix], 1)
				utility_nd[num_INST, h] = np.mean(util/util_ppr)

	print("****************************************************************")
	print("alpha = ",alpha)
	print("Welfare Optimal")
	print(np.mean(overall_opt[1:]))
	print("Welfare obtained")
	print(np.mean(welfare[1:,:],0))
	print("Mean ratio obtained")
	print(np.mean(welfare[1:,:]/np.expand_dims(overall_opt[1:], -1), 0))				
	print("Std ratio obtained")
	print(np.std(welfare[1:,:]/np.expand_dims(overall_opt[1:], -1), 0))
	print("utility of agent 0")
	print(np.mean(utility[1:,:],0), np.std(utility[1:,:],0))
	print(np.mean(utility_nd[1:,:],0), np.std(utility_nd[1:,:],0))
	return overall_opt, welfare, utility, utility_nd
