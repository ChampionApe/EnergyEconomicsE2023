from base import *
from lpCompiler import _blocks
from lpModels import modelShell

# Functions for all mBasicInt models:
def fuelCost(db):
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

def variableCosts(db):
	""" short run costs in 1000€. """
	return db['mc']*pyDbs.pdSum(db['Generation'], 'h') / 1000

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
	return ( pyDbs.pdSum(model.db['Generation'], 'h')/ pdNonZero(pyDbs.pdSum(model.hourlyGeneratingCapacity, 'h')) ).dropna()

def marginalSystemCosts(db):
	return adj.rc_pd(db['λ_equilibrium'], alias={'h_alias':'h'}).droplevel('_type')

def meanMarginalSystemCost(db, var):
	return pyDbs.pdSum( (var * marginalSystemCosts(db)) / pdNonZero(pyDbs.pdSum(var, 'h')), 'h')

def downlift(db):
	return meanMarginalSystemCost(db, db['HourlyDemand']) - meanMarginalSystemCost(db, db['Generation'])

def marginalEconomicRevenue(model):
	ϑ = model.db['λ_Generation'].xs('u', level = '_type')
	ϑ = ϑ[ϑ!=0]
	return pyDbs.pdSum(marginalSystemCosts(model.db) * adj.rc_pd(model.hourlyCapFactors, ϑ), 'h')

def marginalEconomicValue(model):
	return - pyDbs.pdSum(model.db['λ_Generation'].xs('u',level='_type') * model.hourlyCapFactors, 'h').add( 1000 * model.db['FOM'] * len(model.db['h'])/8760, fill_value = 0)

class mSimple(modelShell):
	def __init__(self, db, blocks = None, **kwargs):
		db.updateAlias(alias = [('h','h_alias')])
		super().__init__(db, blocks = blocks, **kwargs)

	@property
	def hourlyGeneratingCapacity(self):
		return (adjMultiIndex.bc(self.db['GeneratingCapacity'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt')

	@property
	def hourlyCapFactors(self):
		return adjMultiIndex.bc(adj.rc_pd(self.db['CapVariation'], self.db['id2hvt']), self.db['id2hvt']).droplevel('hvt')

	@property
	def hourlyLoad_c(self):
		return self.db['LoadVariation'] * self.db['Load']

	@property
	def hourlyLoad(self):
		return pyDbs.pdSum(self.hourlyLoad_c, 'c')

	def preSolve(self, recomputeMC=False, **kwargs):
		if ('mc' not in self.db.symbols) or recomputeMC:
			self.db['mc'] = mc(self.db)

	@property
	def globalDomains(self):
		return {'Generation': pd.MultiIndex.from_product([self.db['h'], self.db['id']]),
				'HourlyDemand': self.db['h'],
				'equilibrium': self.db['h_alias']}
	@property
	def c(self):
		return [{'varName': 'Generation', 'value': adjMultiIndex.bc(self.db['mc'], self.db['h'])},
				{'varName': 'HourlyDemand', 'value': -self.db['MWP']}]
	@property
	def u(self):
		return [{'varName': 'Generation', 'value': self.hourlyGeneratingCapacity},
				{'varName': 'HourlyDemand', 'value': self.hourlyLoad}]
	@property
	def b_eq(self):
		return [{'constrName': 'equilibrium'}]
	@property
	def A_eq(self):
		return [{'constrName': 'equilibrium', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), 'h','h_alias')},
				{'constrName': 'equilibrium', 'varName': 'HourlyDemand', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['HourlyDemand']), 'h','h_alias')}]

	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	def postSolve(self, solution, **kwargs):
		if solution['status'] == 0:
			self.unloadToDb(solution)
			self.db['Welfare'] = -solution['fun']
			self.db['FuelConsumption'] = fuelConsumption(self.db)
			self.db['Emissions'] = emissionsFuel(self.db)
			self.db['capacityFactor'] = theoreticalCapacityFactor(self.db)
			self.db['capacityCosts'] = averageCapacityCosts(self.db)
			self.db['energyCosts'] = averageEnergyCosts(self.db)
			self.db['marginalSystemCosts'] = marginalSystemCosts(self.db)
			self.db['marginalEconomicValue'] = marginalEconomicValue(self)

class mEmissionCap(mSimple):
	def __init__(self, db, blocks=None, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)
	@property
	def b_ub(self):
		return [{'constrName': 'emissionsCap', 'value': self.db['CO2Cap']}]
	@property
	def A_ub(self):
		return [{'constrName': 'emissionsCap', 'varName': 'Generation', 'value': adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.db['h'])}]

class mRES(mSimple):
	def __init__(self, db, blocks=None, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def cleanIds(self):
		s = (self.db['FuelMix'] * self.db['EmissionIntensity']).groupby('id').sum()
		return s[s <= 0].index

	@property
	def b_ub(self):
		return [{'constrName': 'RESCapConstraint'}]
	@property
	def A_ub(self):
		return [{'constrName': 'RESCapConstraint', 'varName': 'Generation', 'value': -1, 'conditions': self.cleanIds},
				{'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand', 'value': self.db['RESCap']}]

class mMultipleConsumers(mSimple):
	def __init__(self, db, blocks=None, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def globalDomains(self):
		return {'Generation': pd.MultiIndex.from_product([self.db['h'], self.db['id']]),
				'HourlyDemand': pyDbs.cartesianProductIndex([self.db['c'], self.db['h']]),
				'equilibrium': self.db['h_alias']}

	@property
	def c(self):
		return [{'varName': 'Generation', 'value': adjMultiIndex.bc(self.db['mc'], self.db['h'])},
				{'varName': 'HourlyDemand', 'value': -adjMultiIndex.bc(self.db['MWP'], self.globalDomains['HourlyDemand'])}]
	@property
	def u(self):
		return [{'varName': 'Generation', 'value': self.hourlyGeneratingCapacity},
				{'varName': 'HourlyDemand', 'value': self.hourlyLoad_c}]
