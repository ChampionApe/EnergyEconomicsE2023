from base import *
from nlModels import *
from scipy.stats import norm

# Functions for all mBasicInt_NonLinear models that never depend on endogenous variables

# def fixedCosts(db):
# 	""" fixed operating and maintenance costs of installed capacity in 1000€. """
# 	return db['FOM']*db['GeneratingCapacity'] * len(db['h'])/8760

# def marginalGeneration(x,model):
# 	return model.db['hourlyGeneratingCapacity'].mul(model.CapacityUtilizationDiscrete(x).transform(norm.pdf).div(model.db['sigma']))

# def AvgExcessDemand(x,model):
# 	return model.ED(x).mean()

# def AvgMarginalFuelConsumption(x,model):
# 	return pyDbs.pdSum(marginalGeneration(x,model).mul(model.db['FuelMix']),'id').groupby('BFt').mean()

# def FuelMixShare(model):
# 	fm_share = model.db['FuelMix'].div(pdNonZero(pyDbs.pdSum(model.db['FuelMix'],['BFt']))).fillna(0)
# 	NoFuel = 1-pyDbs.pdSum(fm_share,'BFt')
# 	return pd.concat([pd.Series(NoFuel.values,index=pd.MultiIndex.from_tuples([('NoFuel',ids) for ids in NoFuel.index],names=['BFt','id'])),fm_share],aidx_xs=0).sort_index().rename('FuelMiXShare')

# def AvgMarginalEnergyShare(x,model):
# 	return pyDbs.pdSum(marginalGeneration(x,model).drop('EDP',level='id',errors='ignore').mul(FuelMixShare(model).drop('EDP',level='id',errors='ignore')),'id').groupby('BFt').mean().div(pyDbs.pdSum(marginalGeneration(x,model).drop('EDP',level='id',errors='ignore'),'id').mean())

# def AvgMarginalEmissions(x,model):
# 	return pyDbs.pdSum(AvgMarginalFuelConsumption(x,model).mul(model.db['EmissionIntensity']),['h','id','BFt'])

class WritePropertyError(Exception):
    pass

class mSimpleNL(modelShellNL):  
	def __init__(self, db, **kwargs):
		mSimpleNL.updateDB(db)
		super().__init__(db,**kwargs)
		self.x_vars = {'endo_var':['p'],'theta_var':['FuelMix']}
		self.idx_x = {
				'p': range(0,len(self.db['h'])),
				'FuelMix': range(len(self.db['h'])+1,len(self.db['h'])+1+len(self.db['FuelMix']))
			} 		# index of vector with endogenous variables
		self.set_model_properties(kwargs)					# initializing model properties
		self.set_model_structure()
		self.set_model_parameters()							# intializing model parameters independent of the market equilibrium

###########################################
# Initialize class
###########################################

	@staticmethod
	def updateDB(db, recomputeMC=False, **kwargs):
		""" Function for initializing model database """
		db.updateAlias(alias = [('h','h_alias')])
		# db.__setitem__('sigma',10); 						# Here we include a smoothing parameter as a scalar
		db['p'] = pd.Series(0,index=db['h'],name='p')		# Here we initialize the price vector

	def set_model_properties(self,kwargs):
		""" Determines the structure of the model """
		if 'model_type' in kwargs.keys():
				if kwargs['model_type'] in ['normal','no machine zeros']:
					self._model_type = kwargs['model_type']
				else:
					raise WritePropertyError("Model type is a string variable of types ['normal','no machine zeros']")
		else:
			self._model_type = 'normal'
		self._sigma = 10
		self.HourlyGeneration = self.simple_HourlyGeneration if self.model_type=='normal' else self.pg_HourlyGeneration if self.model_type=='no machine zeros' else None
		self.dHourlyGeneration_dp = self.simple_dHourlyGeneration_dp if self.model_type=='normal' else self.nzmg_dHourlyGeneration_dp if self.model_type=='no machine zeros' else None
		self.Jacobian_of_ExcessDemand = self.ExcessDemand_Jacobian 
		self.NonLinearSystem = self.ExcessDemand

	def set_model_parameters(self):
		""" This is a function loading in multiple read(-write) model properties (i.e. parameters) """
		self._hourlyGeneratingCapacity = (adjMultiIndex.bc(self.db['GeneratingCapacity'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt').astype(float)
		self._hourlyDemand_c = (self.db['LoadVariation'] * self.db['Load']).astype(float)
		self._Demand = self.hourlyDemand_c.groupby('h').sum()
		self._fuelCost = self.db['FuelPrice'].add(pyDbs.pdSum(self.db['EmissionIntensity'] * self.db['EmissionTax'], 'EmissionType'), fill_value=0).astype(float)
		self._averageMC = (pyDbs.pdSum((self.db['FuelMix'] * self.fuelCost).dropna(), 'BFt') + self.db['OtherMC']).astype(float)
		self._mc_lo = pd.Series(0,index=self.id2h).add(self.averageMC + norm.ppf(np.finfo(float).eps) * self.sigma).astype(float)
		self._mc_nonZeroMarginalGeneration_invpdf = pd.Series(0,index=self.id2h).add(np.finfo(float).eps*self.sigma*np.sqrt(2*np.pi)).div(self.hourlyGeneratingCapacity).apply(np.log).mul(-2).apply(np.sqrt)
		self._mc_nonZeroMarginalGeneration_lo = self.averageMC - self.sigma*self.mc_nonZeroMarginalGeneration_invpdf
		self._mc_nonZeroMarginalGeneration_up = self.averageMC + self.sigma*self.mc_nonZeroMarginalGeneration_invpdf
		self._Targets = None
	
	def set_model_structure(self):
		""" This is a function loading in multiple read model properties """
		self._H = len(self.db['h'])
		self._id2h = (adjMultiIndex.bc(self.db['GeneratingCapacity'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt').index
		self._h2h_alias = pd.MultiIndex.from_tuples([(h,h) for h in self.db['h']],names=['h','h_alias'])
		self._sparse_ones_H_vector = sparse.coo_matrix((np.ones(self.H),(self.db['h'].values-1,np.repeat(0,self.H))),shape=(self.H,1)).tocsc()
		self._h_rowidx = pd.Series(1,index=self.h2h_alias).groupby('h').ngroup().values
		self._h_colidx = pd.Series(1,index=self.h2h_alias).groupby('h_alias').ngroup().values
		self._sparse_ones_HxH_matrix = sparseMatrixFromSeries(pd.Series(1,index=self.h2h_alias).astype(float),columns=['h_alias']).tocsc()


	@property
	def model_type(self):
		return self._model_type

	@model_type.setter
	def model_type(self, _str):
		self._model_type = _str
		self.update_model_equations(_str)

	def update_model_equations(self,_str):
		self.HourlyGeneration = self.simple_HourlyGeneration if _str=='normal' else self.pg_HourlyGeneration if _str=='no machine zeros' else None
		self.dHourlyGeneration_dp = self.simple_dHourlyGeneration_dp if _str=='normal' else self.nzmg_dHourlyGeneration_dp if _str=='no machine zeros' else None
		
	def postSolve(self):
		""" Define some variables to be saved post solution to the model. """
		# self.db['FuelConsumption'] = self.fuelConsumption(self.x)
		# self.db['Emissions'] = self.emissions(self.x)
		self.db['hourlyGeneration'] = self.HourlyGeneration(self.x)
		# self.db['Welfare'] = self.Welfare(self.x)

###########################################
# Set the parameters of the model 
# independent of the market equilibrium
###########################################

	@property
	def sigma(self):
		return self._sigma

	@sigma.setter
	def sigma(self,value):
		self._sigma = value
		self.set_model_parameters()

	@property
	def hourlyGeneratingCapacity(self):
		""" Hourly generating capacity measured in GJ """
		return self._hourlyGeneratingCapacity

	@hourlyGeneratingCapacity.setter
	def hourlyGeneratingCapacity(self,series):
		self._hourlyGeneratingCapacity = series
		self._id2h = series.index
	
	@property
	def hourlyDemand_c(self):
		""" Hourly demand measured in GJ """
		return self._hourlyDemand_c

	@hourlyDemand_c.setter
	def hourlyDemand_c(self,value):
		self._hourlyDemand_c = series

	@property
	def Demand(self):
		""" Hourly demand measured in GJ """
		return self._Demand

	@Demand.setter
	def Demand(self,value):
		raise WritePropertyError('Demand is read-only and given by hourlyDemand_c.')

	@property
	def fuelCost(self):
		""" Marginal fuel costs in €/GJ """
		return self._fuelCost

	@fuelCost.setter
	def fuelCost(self,series):
		self._fuelCost = series

	@property
	def averageMC(self):
		""" Marginal costs in €/GJ """
		return self._averageMC

	@averageMC.setter
	def averageMC(self,series):
		self._averageMC = series

	@property
	def Targets(self):
		return self._Targets

	@Targets.setter
	def Targets(self,array):
		if isinstance(array,np.ndarray):
			self._Targets = array
		else:
			raise WritePropertyError('Targets can only be represented by a numpy array of floats')


###########################################
# Set the indices of the model, 
# determined by the data
###########################################

	@property 
	def H(self):
		return self._H

	@H.setter
	def H(self,multiindex):
		raise WritePropertyError('The number of hours (H) is read-only and determined by the data.')

	@property
	def id2h(self):
		return self._id2h

	@id2h.setter
	def id2h(self,series):
		raise WritePropertyError('The mapping between id and h is always given by the hourly generating capacity.')

	@property 
	def h2h_alias(self):
		return self._h2h_alias

	@h2h_alias.setter
	def h2h_alias(self,multiindex):
		self._h2h_alias = multiindex

###########################################
# Model equations
###########################################

	def xArray2pdSeries(self,x,variable='p'):
		""" Method for transforming endogenous variables from a numpy array to an indexed Pandas Series """
		return pd.Series(x[self.idx_x[variable]],index=self.db[variable].index,name=variable) if variable in self.idx_x.keys() else self.db[variable]

	# Model equations
	def simple_CapacityUtilizationDiscrete(self,x):
		""" Inner object in optimal capacity utilization """
		return pd.Series(0,index=self.id2h).add(self.xArray2pdSeries(x,variable='p')).sub(self.averageMC).div(self.sigma)

	def simple_CapacityUtilization(self,x):
		""" Optimal capacity utilization """
		return self.simple_CapacityUtilizationDiscrete(x).apply(norm.cdf)

	def simple_HourlyGeneration(self,x):
		""" Optimal generation """
		return self.hourlyGeneratingCapacity * self.simple_CapacityUtilization(x)

	def Supply(self,x):
		""" Aggregate supply """
		return self.HourlyGeneration(x).groupby('h').sum()

	def ExcessDemand(self,x):
		""" Equilibrium definition defined as excess demand"""
		return self.Demand - self.Supply(x)

###########################################
# Model equations without machine zeros
###########################################

	@property
	def mc_lo(self):
		""" Below this threshold for marginal costs, the marginal generation is virtually unchanged """
		return self._mc_lo

	@mc_lo.setter
	def mc_lo(self,series):
		raise WritePropertyError("'mc_lo' is read-only and determined by the model parameters.")

	def prices_PositiveGeneration(self,x):
		""" Combinations of [id,h] of hourly prices in which generation is strictly positive """
		p = pd.Series(0,index=self.id2h,name='p').add(self.xArray2pdSeries(x,variable='p'))
		return p[ p>=self.mc_lo ]

	def pg_CapacityUtilizationDiscrete(self,x):
		""" A sparse version of the discrete capacity utilization decision """
		return (( self.prices_PositiveGeneration(x) - self.averageMC ) / self.sigma ).dropna()

	def pg_CapacityUtilization(self,x):
		""" A sparse version of optimal capacity utilization """
		return self.pg_CapacityUtilizationDiscrete(x).apply(norm.cdf)

	def pg_HourlyGeneration(self,x):
		""" A sparse version of optimal generation """
		u = self.pg_CapacityUtilization(x)
		return u.mul(self.hourlyGeneratingCapacity[self.hourlyGeneratingCapacity.index.isin(u.index)])

###########################################
# Jacobian matrix of the Excess Demand
# function
###########################################

	def simple_dCapacityUtilization_dp(self,x):
		""" Derivative of optimal capacity utilization wrt. prices """
		return self.simple_CapacityUtilizationDiscrete(x).apply(norm.pdf) / self.sigma

	def simple_dHourlyGeneration_dp(self,x):
		""" Derivative of optimal generation wrt. prices """
		return self.hourlyGeneratingCapacity * self.simple_dCapacityUtilization_dp(x)

	def dSupply_dp(self,x):
		""" Derivative of aggregate supply wrt. prices """
		return self.dHourlyGeneration_dp(x).groupby('h').sum()

	def dExcessDemand_dp(self,x):
		""" derivative of the excess demand function """
		return -pd.Series(self.dSupply_dp(x).values,index=self.h2h_alias)

	def ExcessDemand_Jacobian(self,x):
		""" Jacobian of the excess demand function """
		out = np.zeros((self.H,self.H)) # matrix of m x m of zeros
		idx = np.diag_indices(self.H,ndim=2) # getting the indices of the diagonal (only diagonal is non-zero)
		out[idx] = self.dExcessDemand_dp(x) # derivatives inserted on the diagonal
		return out

###########################################
# Jacobian matrix of the Excess Demand
# function without machine zeros
###########################################

	@property
	def mc_nonZeroMarginalGeneration_invpdf(self):
		return self._mc_nonZeroMarginalGeneration_invpdf

	@mc_nonZeroMarginalGeneration_invpdf.setter
	def mc_nonZeroMarginalGeneration_invpdf(self):
		raise WritePropertyError("'mc_nonZeroMarginalGeneration_invpdf' is read-only and determined by the model parameters.")

	@property
	def mc_nonZeroMarginalGeneration_lo(self):
		return self._mc_nonZeroMarginalGeneration_lo

	@mc_nonZeroMarginalGeneration_lo.setter
	def mc_nonZeroMarginalGeneration_lo(self):
		raise WritePropertyError("'mc_nonZeroMarginalGeneration_lo' is read-only and determined by the model parameters.")

	@property
	def mc_nonZeroMarginalGeneration_up(self):
		return self._mc_nonZeroMarginalGeneration_up

	@mc_nonZeroMarginalGeneration_up.setter
	def mc_nonZeroMarginalGeneration_up(self):
		raise WritePropertyError("'mc_nonZeroMarginalGeneration_up' is read-only and determined by the model parameters.")

	def prices_nonZeroMarginalGeneration(self,x):
		""" Combinations of [id,h] of hourly prices in which marginal generation is non-zero. """
		p = pd.Series(0,index=self.id2h,name='p').add(self.xArray2pdSeries(x,variable='p'))
		return p[ (p>=self.mc_nonZeroMarginalGeneration_lo) & (p<=self.mc_nonZeroMarginalGeneration_up) ]

	def nzmg_CapacityUtilizationDiscrete(self,x):
		""" A sparse version of the discrete capacity utilization decision """
		return (( self.prices_nonZeroMarginalGeneration(x) - self.averageMC ) / self.sigma ).dropna()

	def nzmg_dCapacityUtilization_dp(self,x):
		""" Derivative of optimal capacity utilization wrt. prices """
		return self.nzmg_CapacityUtilizationDiscrete(x).apply(norm.pdf) / self.sigma

	def nzmg_dHourlyGeneration_dp(self,x):
		""" Derivative of optimal generation wrt. prices """
		u = self.nzmg_dCapacityUtilization_dp(x)
		return u.mul(self.hourlyGeneratingCapacity[self.hourlyGeneratingCapacity.index.isin(u.index)])

###########################################
# User-defined Newton-Kantorowich solver 
# defined over sparse vectors and matrices
###########################################

	@property
	def sparse_ones_H_vector(self):
	 	return self._sparse_ones_H_vector

	@sparse_ones_H_vector.setter
	def sparse_ones_H_vector(self,array):
		raise WritePropertyError("'sparse_ones_H_vector' is read-only and determined by the model parameters.")

	@property
	def h_rowidx(self):
		return self._h_rowidx

	@h_rowidx.setter
	def h_rowidx(self,array):
		raise WritePropertyError("'h_rowidx' is read-only and determined by the model parameters.")

	@property
	def h_colidx(self):
	 	return self._h_colidx

	@h_colidx.setter
	def h_colidx(self,array):
		raise WritePropertyError("'h_colidx' is read-only and determined by the model parameters.")

	@property
	def sparse_ones_HxH_matrix(self):
	 	return self._sparse_ones_HxH_matrix

	@sparse_ones_HxH_matrix.setter
	def sparse_ones_HxH_matrix(self,array):
		raise WritePropertyError("'sparse_ones_HxH_matrix' is read-only and determined by the model parameters.")

	def sparseExcessDemand(self,x):
		""" Sparse vector of the excess demand function.
			NOTE: This function is only used, when a sparse version of the Newton-Kantorowich method is used for solving the model. """
		out = self.sparse_ones_H_vector.copy()
		out[self.h_rowidx,0] = self.ExcessDemand(x).values
		return out

	def ExcessDemand_sparseJacobian(self,x):
		""" sparse Jacobian of the excess demand function """
		# return sparseMatrixFromSeries(self.dED_dp(x),columns=['h_alias']).tocsc()
		out = self.sparse_ones_HxH_matrix.copy()
		out[self.h_rowidx,self.h_colidx] = self.dExcessDemand_dp(x).values
		return out

	def ExcessDemand_sparseInverseJacobian(self,x):
		""" the inverse of the sparse Jacobian of the excess demand function """
		# return sparse.linalg.inv(self.ExcessDemand_sparseJacobian(x))
		out = self.sparse_ones_HxH_matrix.copy()
		out[self.h_rowidx,self.h_colidx] = 1/self.dExcessDemand_dp(x).values
		return out

	def LipSchitzConstant(self,p):
		return sparse.linalg.norm(self.ED_sparseJacobian(p))

	def KantorowichConstant(self,p):
		""" Parameter measuring the rate at which, the solver is approaching the solution (<0.5 is good)"""
		return self.LipSchitzConstant(p) * sparse.linalg.norm(self.ED_sparseInverseJacobian(p)) * np.linalg.norm(self.NewtonStep(p))

	def NewtonStep(self,p):
		""" This function defines the Newton step in Newton-Kantorowich solver. """
		return -(self.ExcessDemand_sparseInverseJacobian(p).dot(self.sparseExcessDemand(p))).toarray().reshape(p.shape)
		# return -(self.ExcessDemand(p)/self.dExcessDemand_dp(p).droplevel('h_alias')).values

	def manualSolver(self,x0=None,n_iter = 100):
		""" Method using Newton-Kantorowich for rootfinding using a sparse Jacobian of the excess demand function """
		i = 1
		p = x0[self.idx_x['p']]
		p_step = np.ones(p.shape)
		while (np.isclose(p_step,b=0).all()==False) & (i<=n_iter):
		# while (np.isclose(self.ExcessDemand(p),b=0).all()==False) & (i<=n_iter):
			# print('At iteration:' '\t' + str(i))
			p_step = self.NewtonStep(p)
			p = p + p_step
			# print('\t The number of markets in equilibrium is:\t' +str(np.isclose(p_step,b=0).sum()))
			# print('\t The Kantorowich constant is:\t' + str(round(self.KantorowichConstant(p),2)))
			i+=1
		return p

###########################################
# Estimation/calibration
###########################################

	def FuelConsumption(self,x):
		return (self.HourlyGeneration(x) * self.db['FuelMix']).groupby(['h','BFt']).sum()


	def averageFuelIntensity(self,x):
		return adjMultiIndex.bc(self.FuelConsumption(x),self.db['h2hDay']).groupby(['hDay','BFt']).sum() / adjMultiIndex.bc(self.Supply(x),self.db['h2hDay']).groupby(['hDay','BFt']).sum().sort_index()

	def MarginalFuelConsumption(self,x):
		return (self.dHourlyGeneration_dp(x) * self.db['FuelMix']).groupby(['h','BFt']).sum()

	def EstimationObjective(self,x,weight_matrix=None):
		diff = self.averageFuelIntensity(x)-self.Targets
		return np.matmul(np.matmul((diff).transpose(),weight_matrix),diff)

###########################################
# Welfare calculations and other
# summary statistics
###########################################
	
	def averageMarginalFuelConsumption(self,x):
		out = self.MarginalFuelConsumption(x).groupby('BFt').mean()
		out.index = pd.MultiIndex.from_tuples([(x[self.idx_x['p']].mean(),k) for k in out.index],names=['p','BFt'])
		return out

	# def Obj(self,x,data,weight_matrix):
	# 	""" Objective function in estimation """
	# 	k = data.shape[0]
	# 	out = np.empty((k,k))
	# 	H = len(self.db['h'])
	# 	# Marginal prices
	# 	out[0:(H-1)] = self.x[self.idx_x['p']]
	# 	# Marginal emissions
	# 	# out[0:(H-1)] = self.marginalEmissions(x)
	# 	return np.matmul(out.transpose(),np.matmul(weight_matrix,out))
	
	# def dObj_dTheta(self,x,data):
	# 	""" derivative of the objective function using the implicit function theorem """
	# 	return 

	# def marginalFuelConsumption(self,x):
	# 	return self.dHourlyGeneration_dp(x).mul(self.db['FuelMix']).groupby('h').sum()

	# @property
	# def marginalEmissions(self,x):
	# 	return pyDbs.pdSum(self.marginalFuelConsumption(x).mul(self.db['EmissionIntensity']),['id','BFt'])


	# # Welfare calculations:
	# def marginalEconomicRevenue(self,x):
	# 	""" marginal economic revenue of generators (i.e. per unit of capacity) """
	# 	return pyDbs.pdSum(self.db['hourlyGeneratingCapacity'].div(self.db['GeneratingCapacity']).mul(self.CapacityUtilization(x)).mul(self.xArray2pdSeries(x,variable='p')), 'h')

	# def marginalEconomicCosts(self,x):
	# 	""" short run optimal costs per generating capacity in 1000€ (i.e. per unit of capacity). """
	# 	u = self.CapacityUtilization(x)
	# 	return pyDbs.pdSum(self.db['hourlyGeneratingCapacity'].div(self.db['GeneratingCapacity']).mul(self.CapacityUtilizationDiscrete(x).transform(norm.pdf)/np.where(u==1,np.finfo(float).eps,1-u)*(self.sigma)),'h')

	# def marginalEconomicValue(self,x):
	# 	return adj.rctree_pd(self.marginalEconomicRevenue(x).sub(self.marginalEconomicCosts(x)),pd.Index([x for x in self.db['id'] if x!='EDP'],name='id'))

	# def ProducerSurplus(self,x):
	# 	return self.marginalEconomicValue(x).mul(adj.rctree_pd(self.db['GeneratingCapacity'],pd.Index([x for x in self.db['id'] if x!='EDP'],name='id')))

	# def WelfareLossFromCapacityScarcity(self,x):
	# 	""" scarcity rents due to insufficient generating capacity in 1000 euro/MWh capacity"""
	# 	return (self.marginalEconomicRevenue(x)-self.marginalEconomicCosts(x)).mul(self.db['GeneratingCapacity']).loc['EDP']

	# def ConsumerSurplus(self,x):
	# 	pdiff = pd.Series(0,index=self.db['hourlyGeneratingCapacity'].index,name='p').add(self.averageMC(x).loc['EDP']).sub(self.xArray2pdSeries(x,variable='p'))
	# 	return pyDbs.pdSum(self.db['hourlyDemand'].mul(pdiff),'h') - self.WelfareLossFromCapacityScarcity(x)

	# def Welfare(self,x):
	# 	""" Economic welfare defined as consumer and producer surplus. """
	# 	return self.marginalEconomicCosts(x).mul(self.db['GeneratingCapacity']).sum()
		# return self.ConsumerSurplus(x).sum() + self.ProducerSurplus(x).sum()
		# return (pd.Series(0,index=self.db['hourlyGeneratingCapacity'].index,name='p').add(self.averageMC(x).loc['EDP']) - self.marginalEconomicCosts(x)).sum()

# class mCapNL(mSimpleNL):
# 	def __init__(self, db, **kwargs):
# 		self.db = self.preSolve(db)
# 		self.idx_x = {
# 			'p': range(0,len(self.db['h'])),
# 			'EmissionTax': len(self.db['h'])
# 		}

# 	def xArray2pdSeries(self,x,variable=None,calibration_variable=None):
# 		""" endogenous variables represented as a pandas series for easier management of dimensions """
# 		# Endogenous market prices:
# 		if variable=='p':
# 			out = pd.Series(x[self.idx_x['p']],index=self.db['h'],name='p')
# 		elif variable=='EmissionTax':
# 			out = pd.Series(x[self.idx_x['EmissionTax']],index=self.db['EmissionTax'].index,name='EmissionTax') if 'EmissionTax' in self.idx_x.keys() else self.db['EmissionTax'].copy()
# 		# Variable for calibrating prices:
# 		# elif variable=='OtherMC':
# 		# 	out = pd.Series(x[self.idx_x['OtherMC']],index=self.db['OtherMC'].index,name='OtherMC') if 'OtherMC' in self.idx_x.keys() else self.db['OtherMC'].copy()
# 		elif isinstance(calibration_variable,str):
# 			out = pd.Series(x[self.idx_x[calibration_variable]],index=self.db[calibration_variable].index,name=calibration_variable) if calibration_variable in self.idx_x.keys() else self.db[calibration_variable].copy()
# 		return out

# 	def CO2Cap(self,x):
# 		""" Equation for CO2 cap """
# 		return self.db['CO2Cap'] - self.emissions(x).loc['CO2']

# 	def NonLinearSystem(self,x):
# 		m = len(self.db['h'])+1
# 		out = np.zeros(m)
# 		out[self.idx_x['p']] = self.ED(x).values
# 		out[self.idx_x['EmissionTax']] = self.CO2Cap(x)
# 		return out

# 	# def maidx_xmizeWelfare(self,x0=None,jacobian=None):
# 	# 	if jacobian is not None:
# 	# 		jacobian = self.dED_dp
# 	# 	sol = minimize(
# 	# 		fun=lambda x: self.Welfare(x), 
# 	# 		x0=x0, 
# 	# 		# constraints = NonlinearConstraint(self.CO2Cap, lb=-np.inf, ub=0), 
# 	# 		constraints = {'type':'ineq', 'fun':self.CO2Cap},
# 	# 		jac=jacobian
# 	# 		# method='trust-constr'
# 	# 	)
# 	# 	if sol['success']:
# 	# 		self.x = sol['x']
# 	# 		self.unloadSolutionToDB(self.x)
# 	# 		self.postSolve()
# 	# 	else:
# 	# 		warnings.warn(r'Solver did not find an maidx_xmize economic welfare.', UserErrorMessage)
# 	# 	return sol['message']

# 	# def objective(self, p, theta, data):
# 	# 	model = np.array([p.mean(),self.generation(p).sum()])
# 	# 	return np.square(model-data).sum()


# 		# sol = minimize(fun=lambda p: (self.excess_demand(p)**2).sum(), x0=p0, jac=jac)
# 		# if sol['success']:
# 		# 	self.db['p'] = pd.Series(sol['x'],index=p0.index,name='p')

# 	# def update_db(self):
# 	# 	if 'p' in self.db.symbols:
# 	# 		self.db['Generation'] = self.fgeneration(self.db['p'])
# 	# 		self.db['ConsumerSurplus'] = ConsumerSurplus(self.db)
# 	# 		self.db['marginalEconomicRevenue'] = marginalEconomicRevenue(self.db)
# 	# 		self.db['marginalEconomicCosts'] = marginalEconomicCosts(self.db)
# 	# 		self.db['marginalEconomicValue'] = marginalEconomicValue(self.db)
# 	# 		self.db['Welfare'] = self.db['ConsumerSurplus'].sum()+self.db['marginalEconomicValue'].sum()
# 	# 	else:
# 	# 		'Database cannot be update (equilibrium prices are not in database).'
	

# 	# def postSolve(self, solution, **kwargs):
# 	# 	if solution['status'] == 0:
# 	# 		self.unloadToDb(solution)
# 	# 		self.db['Welfare'] = -solution['fun']
# 	# 		self.db['FuelConsumption'] = fuelConsumption(self.db)
# 	# 		self.db['Emissions'] = emissionsFuel(self.db)
# 	# 		self.db['capacityFactor'] = theoreticalCapacityFactor(self.db)
# 	# 		self.db['capacityCosts'] = averageCapacityCosts(self.db)
# 	# 		self.db['energyCosts'] = averageEnergyCosts(self.db)
# 	# 		self.db['marginalSystemCosts'] = marginalSystemCosts(self.db)
# 	# 		self.db['marginalEconomicValue'] = marginalEconomicValue(self)