from base import *
from scipy.stats import norm
from scipy.optimize import fsolve, minimize
import numpy as np

# Functions for all mBasicInt_NonLinear models

def fuelCost(db):
	""" Marginal fuel costs in €/GJ """
	return db['FuelPrice'].add(pyDbs.pdSum(db['EmissionIntensity'] * db['EmissionTax'], 'EmissionType'), fill_value=0)

def mc(db):
	""" Marginal costs in €/GJ """
	return pyDbs.pdSum((db['FuelMix'] * fuelCost(db)).dropna(), 'BFt').add(db['OtherMC'])

def fuelConsumption(db):
	return pyDbs.pdSum((db['Generation'] * db['FuelMix']).dropna(), ['h','id'])

def plantEmissionIntensity(db):
	return pyDbs.pdSum(db['FuelMix'] * db['EmissionIntensity'], 'BFt')

def emissionsFuel(db):
	return pyDbs.pdSum(fuelConsumption(db) * db['EmissionIntensity'], 'BFt')

def fixedCosts(db):
	""" fixed operating and maintenance costs of installed capacity in 1000€. """
	return db['FOM']*db['GeneratingCapacity'] * len(db['h'])/8760

def hourlyGeneratingCapacity(db):
	return (adjMultiIndex.bc(db['GeneratingCapacity'], db['id2hvt']) * db['CapVariation']).dropna().droplevel('hvt')

def capacity_utilization_discrete(db):
	""" Inner object in optimal capacity utilization """
	return pd.Series(0,index=db['hourlyGeneratingCapacity'].index).add(db['p']).sub(db['mc']).div(db['sigma'])

def capacity_utilization(db):
	""" capacity utilization """
	return capacity_utilization_discrete(db).transform(norm.cdf)

def marginalEconomicCosts(db):
	""" short run optimal costs per generating capacity in 1000€. """
	u = capacity_utilization(db)
	return pyDbs.pdSum(db['hourlyGeneratingCapacity'].mul(capacity_utilization_discrete(db).transform(norm.pdf)/np.where(u==1,np.finfo(float).eps,u)*(db['sigma'])).div(db['GeneratingCapacity']),'h')

def variableCosts(db):
	""" short run costs in 1000€. """
	return pyDbs.pdSum(db['hourlyGeneratingCapacity']*marginalEconomicCosts(db),'h')/1000

def totalCosts(db):
	""" total electricity generating costs in 1000€ """
	return fixedCosts(db).add(variableCosts(db),fill_value = 0)

def averageCapacityCosts(db):
	return 1000 * totalCosts(db) / pdNonZero(db['GeneratingCapacity'])

def averageEnergyCosts(db):
	return 1000 * totalCosts(db) / pdNonZero(pyDbs.pdSum(db['Generation'], 'h'))

def theoreticalCapacityFactor(db):
	return pyDbs.pdSum( (db['Generation']/pdNonZero(len(db['h']) * db['GeneratingCapacity'])).dropna(), 'h')

def practicalCapacityFactor(model):
	return ( pyDbs.pdSum(model.db['Generation'], 'h')/ pdNonZero(pdSum(hourlyGeneratingCapacity(db), 'h')) ).dropna()

def meanMarginalSystemCost(db, var):
	return pyDbs.pdSum( (var * marginalSystemCosts(db)) / pdNonZero(pyDbs.pdSum(var, 'h')), 'h')

def downlift(db):
	return meanMarginalSystemCost(db, db['HourlyDemand']) - meanMarginalSystemCost(db, db['Generation'])

def WelfareLossFromCapacityScarcity(db):
	""" scarcity rents due to insufficient generating capacity in 1000 euro/MWh capacity"""
	return (marginalEconomicRevenue(db)-marginalEconomicCosts(db)).xs('EDP')

def marginalEconomicRevenue(db):
	return pyDbs.pdSum(db['hourlyGeneratingCapacity'].div(db['GeneratingCapacity']).mul(capacity_utilization(db)).mul(db['p']), 'h')

def marginalEconomicValue(db):
	return adj.rc_pd(marginalEconomicRevenue(db)-marginalEconomicCosts(db),pd.Index([x for x in db['id'] if x!='EDP'],name='id'))
	# return - pyDbs.pdSum(model.db['λ_Generation'].xs('u',level='_type') * model.hourlyCapFactors, 'h').add( 1000 * model.db['FOM'] * len(model.db['h'])/8760, fill_value = 0)

def ConsumerSurplus(db):
	pdiff = pd.Series(0,index=db['hourlyGeneratingCapacity'].index,name='p').add(db['mc'].loc['EDP']).sub(db['p'])
	return pyDbs.pdSum(db['HourlyDemand'].mul(pdiff),'h') - WelfareLossFromCapacityScarcity(db)


class mSimpleNonLinear:    
	def __init__(self, db, **kwargs):
		self.db = self.preSolve(db)

	@staticmethod
	def preSolve(db, recomputeMC=False, **kwargs):
		db.updateAlias(alias = [('h','h_alias')])
		if ('mc' not in db.symbols) or recomputeMC:
			db['mc'] = mc(db)
		if 'hourlyGeneratingCapacity' not in db.symbols:
			db['hourlyGeneratingCapacity'] = hourlyGeneratingCapacity(db)
		if 'HourlyDemand' not in db.symbols:
			db['HourlyDemand'] = db['LoadVariation'].mul(db['Load'])
		if 'sigma' not in db.symbols:
			db['sigma'] = 10
		return db

	def fcapacity_utilization_discrete(self,p):
		""" Inner object in optimal capacity utilization """
		return pd.Series(0,index=self.db['hourlyGeneratingCapacity'].index).add(pd.Series(p,index=self.db['h'])).sub(self.db['mc']).div(self.db['sigma'])

	def fgeneration(self,p):
		""" Optimal generation """
		return self.db['hourlyGeneratingCapacity']*self.fcapacity_utilization_discrete(p).transform(norm.cdf)

	def fsupply(self,p):
		""" Aggregate supply """
		return self.fgeneration(p).groupby('h').sum()

	def fexcess_demand(self,p):
		""" Equilibrium definition defined as excess demand"""
		return self.db['HourlyDemand'].groupby('h').sum() - self.fsupply(p)

	def jacobian(self,p):
		""" Jacobian of the excess demand function """
		m = len(self.db['h']) # dimensions of jacobian
		out = np.zeros((m,m)) # matrix of m x m of zeros
		idx = np.diag_indices(m,ndim=2) # getting the indices of the diagonal (only diagonal is non-zero)
		out[idx] = self.db['hourlyGeneratingCapacity'].mul(self.fcapacity_utilization_discrete(p).transform(norm.pdf).div(self.db['sigma'])).mul(-1).groupby('h').sum() # drivatives inserted on the diagonal
		return out

	def solve(self,p,p0=None,jacobian=None):
		if p0 is None:
			p0 = pd.Series(0,index=self.db['h'],name='p')
		if jacobian is not None:
			jacobian = self.jacobian
		roots = fsolve(lambda p: self.fexcess_demand(p), x0=p0, fprime=jacobian)
		if isinstance(roots,np.ndarray):
			self.db['p'] = pd.Series(roots,index=self.db['h'],name='p')
		# sol = minimize(fun=lambda p: (self.excess_demand(p)**2).sum(), x0=p0, jac=jac)
		# if sol['success']:
		# 	self.db['p'] = pd.Series(sol['x'],index=p0.index,name='p')

	def update_db(self):
		if 'p' in self.db.symbols:
			self.db['Generation'] = self.fgeneration(self.db['p'])
			self.db['ConsumerSurplus'] = ConsumerSurplus(self.db)
			self.db['marginalEconomicRevenue'] = marginalEconomicRevenue(self.db)
			self.db['marginalEconomicCosts'] = marginalEconomicCosts(self.db)
			self.db['marginalEconomicValue'] = marginalEconomicValue(self.db)
			self.db['Welfare'] = self.db['ConsumerSurplus'].sum()+self.db['marginalEconomicValue'].sum()
		else:
			'Database cannot be update (equilibrium prices are not in database).'
		

	# def postSolve(self, solution, **kwargs):
	# 	if solution['status'] == 0:
	# 		self.unloadToDb(solution)
	# 		self.db['Welfare'] = -solution['fun']
	# 		self.db['FuelConsumption'] = fuelConsumption(self.db)
	# 		self.db['Emissions'] = emissionsFuel(self.db)
	# 		self.db['capacityFactor'] = theoreticalCapacityFactor(self.db)
	# 		self.db['capacityCosts'] = averageCapacityCosts(self.db)
	# 		self.db['energyCosts'] = averageEnergyCosts(self.db)
	# 		self.db['marginalSystemCosts'] = marginalSystemCosts(self.db)
	# 		self.db['marginalEconomicValue'] = marginalEconomicValue(self)