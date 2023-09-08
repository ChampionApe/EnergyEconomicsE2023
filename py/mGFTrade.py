from base import *
from lpCompiler import _blocks
from lpModels import modelShell

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
	return (1000 * totalCosts(db) / pdNonZero(db['GeneratingCapacity'])).droplevel('g')

def averageEnergyCosts(db):
	return (1000 * totalCosts(db) / pdNonZero(pyDbs.pdSum(db['Generation'], 'h'))).droplevel('g')

def theoreticalCapacityFactor(db):
	return pyDbs.pdSum( (db['Generation']/pdNonZero(len(db['h']) * db['GeneratingCapacity'])).dropna(), 'h').droplevel('g')

def practicalCapacityFactor(model):
	return ( pyDbs.pdSum(model.db['Generation'], 'h')/ pdNonZero(pyDbs.pdSum(model.hourlyGeneratingCapacity, 'h')) ).dropna().droplevel('g')

def marginalSystemCosts(db):
	return adj.rc_pd(db['λ_equilibrium'], alias={'h_constr':'h', 'g_constr': 'g'}).droplevel('_type')

def meanMarginalSystemCost(db, var):
	return pyDbs.pdSum( (var * marginalSystemCosts(db)) / pdNonZero(pyDbs.pdSum(var, 'h')), 'h')

def downlift(db):
	return meanMarginalSystemCost(db, db['HourlyDemand']) - meanMarginalSystemCost(db, db['Generation'])

def marginalEconomicRevenue(model):
	ϑ = model.db['λ_Generation'].xs('u', level = '_type')
	ϑ = ϑ[ϑ!=0]
	return pyDbs.pdSum(marginalSystemCosts(model.db) * adj.rc_pd(model.hourlyCapFactors, ϑ), 'h')

def marginalEconomicValue(model):
	return (- pyDbs.pdSum(model.db['λ_Generation'].xs('u',level='_type') * model.hourlyCapFactors, 'h').add( 1000 * model.db['FOM'] * len(model.db['h'])/8760, fill_value = 0)).droplevel('g')

def priceDifferences(db):
	pe = adj.rc_pd(db['marginalSystemCosts'], db['Transmission'].index.droplevel('g_alias'))
	return sortAll(pd.Series(0,index=db['gConnected']).add(-pe).add(pe.rename_axis(index={'g':'g_alias'})))

def congestionRent(db):
	return priceDifferences(db) * db['Transmission']

class mSimple(modelShell):
	def __init__(self, db, blocks=None, **kwargs):
		db.updateAlias(alias=[('h','h_constr'), ('g','g_alias'),('g','g_constr'),('id','id_constr')])
		db['gConnected'] = db['lineCapacity'].index
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def hourlyGeneratingCapacity(self):
		return (adjMultiIndex.bc(self.db['GeneratingCapacity'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt')

	@property
	def hourlyCapFactors(self):
		return adjMultiIndex.bc(adj.rc_pd(self.db['CapVariation'], self.db['id2hvt']), self.db['id2hvt']).droplevel('hvt')

	@property
	def hourlyLoad_c(self):
		return adjMultiIndex.bc(self.db['LoadVariation'] * self.db['Load'], self.db['c2g'])

	@property
	def hourlyLoad(self):
		return pyDbs.pdSum(self.hourlyLoad_c, 'c')

	def preSolve(self, recomputeMC=False, **kwargs):
		if ('mc' not in self.db.symbols) or recomputeMC:
			self.db['mc'] = mc(self.db)

	@property
	def globalDomains(self):
		return {'Generation': pyDbs.cartesianProductIndex([self.db['id2g'], self.db['h']]),
				'GeneratingCapacity': self.db['id'],
				'HourlyDemand': pyDbs.cartesianProductIndex([self.db['c2g'], self.db['h']]),
				'equilibrium': pd.MultiIndex.from_product([self.db['g_constr'], self.db['h_constr']]),
				'Transmission': pyDbs.cartesianProductIndex([self.db['gConnected'],self.db['h']]),
				'ECapConstr': pyDbs.cartesianProductIndex([adj.rc_pd(self.db['id2g'], alias = {'id':'id_constr','g':'g_constr'}), self.db['h_constr']]),
				'TechCapConstr': self.db['TechCap'].index}

	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	@property
	def c(self):
		return [{'varName': 'Generation', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['Generation'])},
				{'varName': 'HourlyDemand', 'value': -adjMultiIndex.bc(self.db['MWP'], self.globalDomains['HourlyDemand'])},
				{'varName': 'Transmission', 'value': adjMultiIndex.bc(self.db['lineMC'], self.db['h'])},
				{'varName': 'GeneratingCapacity','value': adjMultiIndex.bc(self.db['InvestCost_A'], self.db['id2tech']).droplevel('tech').add(self.db['FOM'],fill_value=0)*1000*len(self.db['h'])/8760}]

	@property
	def u(self):
		return [{'varName': 'HourlyDemand', 'value': self.hourlyLoad_c},
				{'varName': 'Transmission', 'value': adjMultiIndex.bc(self.db['lineCapacity'], self.db['h'])}]
	@property
	def b_eq(self):
		return [{'constrName': 'equilibrium'}]
	@property
	def A_eq(self):
		return [{'constrName': 'equilibrium', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), ['g','h'],['g_constr','h_constr'])},
				{'constrName': 'equilibrium', 'varName': 'HourlyDemand', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['HourlyDemand']), ['g','h'],['g_constr','h_constr'])},
				{'constrName': 'equilibrium', 'varName': 'Transmission', 'value': [appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Transmission']), ['g','h'],['g_constr','h_constr']),
																				   appIndexWithCopySeries(pd.Series(1-self.db['lineLoss'], index = self.globalDomains['Transmission']), ['g_alias','h'], ['g_constr','h_constr'])]}]

	@property
	def b_ub(self):
		return [{'constrName': 'ECapConstr'}, {'constrName': 'TechCapConstr', 'value': self.db['TechCap']}]
	@property
	def A_ub(self):
		return [{'constrName': 'ECapConstr', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), ['g','h','id'], ['g_constr','h_constr','id_constr'])},
				{'constrName': 'ECapConstr', 'varName': 'GeneratingCapacity', 'value': -appIndexWithCopySeries(adj.rc_AdjPd(adjMultiIndex.bc(self.hourlyCapFactors, self.db['id2g']), alias = {'h':'h_constr', 'g':'g_constr'}), 'id','id_constr')},
				{'constrName': 'TechCapConstr', 'varName': 'GeneratingCapacity', 'value': reduce(adjMultiIndex.applyMult, [pd.Series(1, index = self.globalDomains['GeneratingCapacity']), self.db['id2tech'], self.db['id2g']])}]

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
			self.db['congestionRent'] = congestionRent(self.db)
			self.db['meanConsumerPrice'] = meanMarginalSystemCost(self.db, self.db['HourlyDemand'])


class mEmissionCap(mSimple):
	def __init__(self, db, blocks=None, commonCap = True, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)
		self.commonCap = commonCap

	@property
	def b_ub(self):
		return super().b_ub + [{'constrName': 'emissionsCap', 'value': pyDbs.pdSum(self.db['CO2Cap'], 'g') if self.commonCap else adj.rc_pd(self.db['CO2Cap'], alias = {'g':'g_constr'})}]

	@property
	def A_ub(self):
		return super().A_ub + [{'constrName': 'emissionsCap', 'varName': 'Generation', 
		'value': adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level = 'EmissionType'), self.globalDomains['Generation']) if self.commonCap else appIndexWithCopySeries(adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation']),'g','g_constr')}]

class mRES(mSimple):
	def __init__(self, db, blocks=None, commonCap = True, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)
		self.commonCap = commonCap

	@property
	def cleanIds(self):
		s = (self.db['FuelMix'] * self.db['EmissionIntensity']).groupby('id').sum()
		return s[s <= 0].index

	@property
	def b_ub(self):
		return super().b_ub + [{'constrName': 'RESCapConstraint', 'value': 0 if self.commonCap else adj.rc_pd(pd.Series(0, index = self.db['RESCap'].index), alias = {'g':'g_constr'})}]

	@property
	def A_ub(self):
		if self.commonCap:
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation', 'value': -1, 'conditions': self.cleanIds},
					{'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand', 'value': self.db['RESCap'].mean()}]
		else:
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation']), 'g','g_constr'), 'conditions': self.cleanIds},
					{'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand', 'value': appIndexWithCopySeries(adjMultiIndex.bc(self.db['RESCap'], self.globalDomains['HourlyDemand']), 'g', 'g_constr')}]