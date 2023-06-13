from functools import reduce
from base import *
_blocks = ('c','l','u','b_eq','b_ub','A_eq','A_ub')
_stdLinProg = ('c', 'A_ub','b_ub','A_eq','b_eq','bounds')

class lpBlock:
	def __init__(self, globalDomains=None, **kwargs):
		self.globalDomains=noneInit(globalDomains, {})
		self.parameters = {k: {} for k in _blocks}
		self.compiled = {k: {} for k in _blocks}
		self.denseArgs = dict.fromkeys(_blocks)
		self.gIndex = {}

	def checkGlobalDomains(self, key, value, defaultValue = 0, conditions=None):
		if key in self.globalDomains:
			return adj.rc_pd(pd.Series(noneInit(value, defaultValue), index = self.globalDomains[key]), c = conditions)
		else:
			return noneInit(value, defaultValue)

	def addVector(self, t, func, component, value, name = None, conditions = None):
		if isinstance(value, pd.Series):
			self.parameters[t][(name, component)] = adj.rc_pd(value, c = conditions)
		elif isinstance(value, (int,float,np.generic, type(None))):
			self.parameters[t][(name, component)] =  self.checkGlobalDomains(name, value, conditions=conditions)
		elif is_iterable(value):
			self.parameters[t][(name, component)] = adj.rc_pd(func(value), c = conditions)
		else:
			raise TypeError(f"The argument '({name}, {component})' added to {t}-blocks should be of type pd.Series, scalar, or an iterable object.")

	def addVectorConstraint(self, t, func, value, name = None, conditions = None):
		if isinstance(value, pd.Series):
			self.parameters[t][name] = adj.rc_pd(value, c = conditions)
		elif isinstance(value, (int,float,np.generic, type(None))):
			self.parameters[t][name] =  self.checkGlobalDomains(name, value, conditions = conditions)
		elif is_iterable(value):
			self.parameters[t][name] = adj.rc_pd(func(value), c = conditions)
		else:
			raise TypeError(f"The argument '{name}' added to {t}-blocks should be of type pd.Series, scalar, or an iterable object.")

	def addMatrix(self, t, component, value, varName = None, constrName = None, conditions=None):
		if isinstance(value, pd.Series):
			self.parameters[t][(constrName, varName, component)] = adj.rc_pd(value, c = conditions)
		elif isinstance(value, (int, float, np.generic, type(None))):
			self.parameters[t][(constrName, varName, component)] = adjMultiIndex.bcAdd(self.checkGlobalDomains(constrName, value, conditions = conditions), self.checkGlobalDomains(varName, 0, conditions=conditions))
		elif is_iterable(value):
			self.parameters[t][(constrName, varName, component)] = adj.rc_pd(sumIte(value), c = conditions)
		else:
			raise TypeError(f"The argument '({varName}, {constrName}, {component})' added to {t}-blocks should be of type pd.Series, scalar, or an iterable object.")

	def add_c(self, component=None, value = None, varName = None, conditions=None):
		self.addVector('c',sumIte,component, value, name = varName, conditions=conditions)

	def add_l(self, component=None, value = None, varName = None, conditions=None):
		self.addVector('l',maxIte,component, value, name = varName, conditions=conditions)

	def add_u(self, component=None, value = None, varName = None, conditions=None):
		self.addVector('u',minIte,component, value, name = varName, conditions=conditions)

	def add_b_eq(self, value = None, constrName = None, conditions=None):
		self.addVectorConstraint('b_eq',sumIte, value, name = constrName, conditions=conditions)

	def add_b_ub(self, value = None, constrName = None, conditions=None):
		self.addVectorConstraint('b_ub',sumIte, value, name = constrName, conditions=conditions)

	def add_A_eq(self, component = None, value = None, varName = None, constrName = None, conditions=None):
		self.addMatrix('A_eq', component, value, varName = varName, constrName = constrName, conditions=conditions)

	def add_A_ub(self, component = None, value = None, varName = None, constrName = None, conditions=None):
		self.addMatrix('A_ub', component, value, varName = varName, constrName = constrName, conditions=conditions)

	def compileVector(self, t, func, name, checkTupleIndex = 0):
		self.compiled[t][name] = fIndexVariable(name, func([v for k,v in self.parameters[t].items() if k[checkTupleIndex] == name]))
	def compileVectorConstraint(self, t, name, btype):
		self.compiled[t][name] = fIndexVariable(name, self.parameters[t][name], btype = btype)
	def compileMatrix(self, t, constrName, varName):
		A, b = sumIte([v for k,v in self.parameters[f'A_{t}'].items() if k[0:2] == (constrName, varName)]), self.parameters[f'b_{t}'][constrName]
		overlap = set(pyDbs.getDomains(A)).intersection(pyDbs.getDomains(b))
		onlyA = set(pyDbs.getDomains(A))-overlap
		if not overlap:
			full = adjMultiIndex.bc(fIndexVariable(varName, A), fIndex(constrName, pyDbs.getIndex(b), btype=t))
		else:
			full = adjMultiIndex.bc(A, pyDbs.getIndex(b))
			if not onlyA:
				full.index = pyDbs.cartesianProductIndex([fIndex(varName, None), fIndex(constrName, full.index, btype=t)])
			else:
				f1, f2 = fIndex(varName, full.index.droplevel(list(set(pyDbs.getDomains(A))-onlyA)) ), fIndex(constrName, full.index.droplevel(list(onlyA)), btype=t)
				full.index = pd.MultiIndex.from_arrays(np.concatenate([f1.to_frame(index=False).values, f2.to_frame(index=False)], axis=1).T, names = stdNames('v')+stdNames(t))
		full.index._nA, full.index._nb = sorted(onlyA), sorted(pyDbs.getDomains(b))
		self.compiled[f'A_{t}'][(constrName, varName)] = full

	def compileParameters(self):
		[self.compileVector('c', sumIte, name) for name in set([n[0] for n in self.parameters['c']])];
		[self.compileVector('l', maxIte, name) for name in set([n[0] for n in self.parameters['l']])];
		[self.compileVector('u', minIte, name) for name in set([n[0] for n in self.parameters['u']])];
		[self.compileVectorConstraint(f'b_{t}', name, btype = t) for t in ('eq','ub') for name in self.parameters[f'b_{t}']];
		[self.compileMatrix(t, tup[0], tup[1]) for t in ('eq','ub') for tup in set([k[0:2] for k in self.parameters[f'A_{t}']])];
	
	def settingsFromCompiled(self):
		self.allvars = sorted(self.getVariables)
		self.allconstr = {t: sorted(self.compiled[f'b_{t}']) for t in ('eq','ub')}
		self.alldomains = ( {k:v.index._n for k,v in reduce(lambda x,y: x|y, [self.compiled[k] for k in ('c','l','u')]).items()} | 
							{k[1]: v.index._nA for k,v in self.compiled['A_eq'].items()} |
							{k[1]: v.index._nA for k,v in self.compiled['A_ub'].items()} )
		self.allconstrdomains = {k: self.compiled[f'b_{t}'][k].index._n for t in ('eq','ub') for k in self.allconstr[t]}

	@property
	def getVariables(self):
		return set([k[1] for l in [self.compiled['A_eq'],self.compiled['A_ub']] for k in l]).union(set.union(*[set(self.compiled[k]) for k in ('c','l','u')]))

	def variableDomains(self, k):
		index = reduce(pd.Index.union, ([self.compiled[t][k].index.levels[1] for t in ('c','l','u') if k in self.compiled[t]]+[self.compiled[t][v].index.levels[1] for t in ('A_eq','A_ub') for v in self.compiled[t] if v[1] == k]))
		return None if index.empty else index

	# Infer global index from compiled parameters
	def inferGlobalDomains(self):
		self.gIndex = {k: fIndex(k, self.variableDomains(k)) for k in self.allvars}
		self.globalVariableIndex = stackIndex(self.gIndex.values(), names = stdNames('v'))
		self.globalConstraintIndex = {t: stackIndex([self.compiled[f'b_{t}'][k] for k in self.allconstr[t]], names = stdNames(t)) if self.compiled[f'b_{t}'] else None for t in ('eq','ub')}
		self.globalMaps = ({'v': pd.Series(range(len(self.globalVariableIndex)), index = self.globalVariableIndex)} | 
							{t: pd.Series(range(len(self.globalConstraintIndex[t])), index = self.globalConstraintIndex[t]) if self.compiled[f'b_{t}'] else None for t in ('eq','ub')}
						)

	def getDenseArgs(self):
		""" NOTE: Vectors are broadcasted """
		[self.denseArgs.__setitem__(t, stackSeries([self.broadcastAndSort_i(t,k,defaultValue=0) for k in self.allvars], names = stdNames('v'))) for t in ('c','l')];
		[self.denseArgs.__setitem__('u', stackSeries([self.broadcastAndSort_i('u',k,defaultValue=None) for k in self.allvars], names = stdNames('v')))];
		[self.denseArgs.__setitem__(f'b_{t}', stackSeries([self.compiled[f'b_{t}'][k] for k in self.allconstr[t]], names = stdNames(t)) if self.allconstr[t] else None) for t in ('eq','ub')];
		[self.denseArgs.__setitem__(f'A_{t}', self.getDenseA(t) if self.allconstr[t] else None) for t in ('eq','ub')];

	def getDenseA(self, t):
		return stackSeries([self.compiled[f'A_{t}'][(constr,var)] for var in self.allvars for constr in self.allconstr[t] if (constr,var) in self.compiled[f'A_{t}']], names = stdNames('v')+stdNames(t))
	def columnIndexFromA(self, Avector, t):
		return Avector.droplevel(stdNames(t)).index.map(self.globalMaps['v'])
	def rowIndexFromA(self, Avector, t):
		return Avector.droplevel(stdNames('v')).index.map(self.globalMaps[t])
	def broadcastAndSort_i(self, t, k, defaultValue = 0):
		return adjMultiIndex.bc(self.compiled[t][k] if k in self.compiled[t] else defaultValue, self.gIndex[k], fill_value = defaultValue).sort_index()


	# 5: Methods to get the stacked numpy arrays:
	def __call__(self, execute=None):
		[getattr(self, k)() for k in noneInit(execute, ['compileParameters','settingsFromCompiled','inferGlobalDomains','getDenseArgs'])];
		return self.lp_args

	@property
	def lp_args(self):
		return {k: getattr(self, 'lp_'+k) for k in _stdLinProg}
	@property
	def lp_c(self):
		return self.denseArgs['c'].values
	@property
	def lp_l(self): 
		return self.denseArgs['l'].values
	@property
	def lp_u(self): 
		return self.denseArgs['u'].values
	@property
	def lp_bounds(self):
		return np.vstack([self.lp_l, self.lp_u]).T
	@property
	def lp_A_eq(self):
		return sparse.coo_matrix((self.denseArgs['A_eq'].values, (self.rowIndexFromA(self.denseArgs['A_eq'],'eq'), self.columnIndexFromA(self.denseArgs['A_eq'],'eq'))), shape = (len(self.globalConstraintIndex['eq']), len(self.globalVariableIndex))) if self.allconstr['eq'] else None
	@property
	def lp_A_ub(self):
		return sparse.coo_matrix((self.denseArgs['A_ub'].values, (self.rowIndexFromA(self.denseArgs['A_ub'],'ub'), self.columnIndexFromA(self.denseArgs['A_ub'],'ub'))), shape = (len(self.globalConstraintIndex['ub']), len(self.globalVariableIndex))) if self.allconstr['ub'] else None
	@property
	def lp_b_eq(self):
		return self.denseArgs['b_eq'].values if self.allconstr['eq'] else None
	@property
	def lp_b_ub(self):
		return self.denseArgs['b_ub'].values if self.allconstr['ub'] else None


	# CHECKING LOWER/UPPER BOUNDS: Returns index where lower and upper bounds are equal.
	# In these instances, the solver does not distinguish between the two, and may ascribe the dual variable to either.
	# The following two functions ascribe the dual variable to the UPPER bound, if the two bounds are the same (scalar bounds)
	def scalarDualLower(self, sol):
		return np.where(self.lp_bounds[:,0]==self.lp_bounds[:,1], 0, sol['lower']['marginals'])

	def scalarDualUpper(self, sol):
		return np.where(self.lp_bounds[:,0]==self.lp_bounds[:,1], np.add(sol['lower']['marginals'], sol['upper']['marginals']), sol['upper']['marginals'])

	# # Get dual solutions:
	# def dual_solution(self, sol, scalarDual = True):
	# 	return pd.Series(self.dual_solutionValues(sol, scalarDual=scalarDual), index = self.dual_solutionIndex)

	@property
	def dualIndex(self):
		lenNone = lambda x: 0 if x is None else len(x)
		ite = (self.globalConstraintIndex['eq'], self.globalConstraintIndex['ub'], self.globalVariableIndex, self.globalVariableIndex)
		return pd.MultiIndex.from_frame(
				pd.MultiIndex.from_tuples(np.hstack([i.values for i in ite if i is not None]),
											names = stdNames('s')).to_frame(index=False).assign(_type = np.hstack([['eq']*lenNone(ite[0]), ['ub']*lenNone(ite[1]), ['l']*len(ite[2]), ['u']*len(ite[3])]))
				)

	def dualValues(self, sol, scalarDual=True):
		if scalarDual:
			return np.hstack([sol['eqlin']['marginals'], sol['ineqlin']['marginals'], self.scalarDualLower(sol), self.scalarDualUpper(sol)])
		else:
			return np.hstack([sol['eqlin']['marginals'], sol['ineqlin']['marginals'], sol['lower']['marginals'], sol['upper']['marginals']])

	# Get dual solutions:
	def dualSolution(self, sol, scalarDual = True):
		return pd.Series(self.dualValues(sol, scalarDual=scalarDual), index = self.dualIndex)
