from base import *
from lpCompiler import _blocks
from lpModels import modelShell

def adHocMerge(x,y):
	return pd.merge(x.rename('x'), pd.Series(0, index = y).rename('y'), left_index = True, right_index = True).dropna().sum(axis=1)

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

def marginalSystemCosts(db):
	return -adj.rc_pd(db['λ_equilibrium'], alias={'h_alias':'h', 'g_alias2': 'g'}).droplevel('_type')

def meanMarginalSystemCost(db, var):
	return pyDbs.pdSum( (var * marginalSystemCosts(db)) / pdNonZero(pyDbs.pdSum(var, 'h')), 'h')

def downlift(db):
	return meanMarginalSystemCost(db, db['HourlyDemand']) - meanMarginalSystemCost(db, db['Generation'])

def priceDifferences(db):
	pe = adj.rc_pd(db['marginalSystemCosts'], db['Transmission'].index.droplevel('g_alias'))
	return sortAll(pd.Series(0,index=db['gConnected']).add(-pe).add(pe.rename_axis(index={'g':'g_alias'})))

def congestionRent(db):
	return priceDifferences(db) * db['Transmission']

class mSimple(modelShell):
	def __init__(self, db, blocks=None, **kwargs):
		db.updateAlias(alias=[('h','h_alias'), ('g','g_alias'),('g','g_alias2'),('id','id_alias')])
		db['gConnected'] = db['lineCapacity'].index
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def hourlyGeneratingCapacity(self):
		return (adjMultiIndex.bc(self.db['GeneratingCapacity'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt')

	@property
	def hourlyCapFactors(self):
		return adjMultiIndex.bc(adj.rc_pd(self.db['CapVariation'], self.db['id2hvt']), self.db['id2hvt']).droplevel('hvt')

	@property
	def hourlyLoad(self):
		return pyDbs.pdSum(adjMultiIndex.bc(self.db['Load'] * self.db['LoadVariation'], self.db['c2g']), 'c')

	def preSolve(self, recomputeMC=False, **kwargs):
		if ('mc' not in self.db.symbols) or recomputeMC:
			self.db['mc'] = mc(self.db)

	@property
	def globalDomains(self):
		return {'Generation': pyDbs.cartesianProductIndex([self.db['id2g'], self.db['h']]),
				'GeneratingCapacity': self.db['id'],
				'HourlyDemand': pd.MultiIndex.from_product([self.db['g'], self.db['h']]),
				'equilibrium': pd.MultiIndex.from_product([self.db['g_alias2'], self.db['h_alias']]),
				'Transmission': pyDbs.cartesianProductIndex([self.db['gConnected'],self.db['h']]),
				'ECapConstr': pyDbs.cartesianProductIndex([adj.rc_AdjPd(self.db['id2g'], alias={'id':'id_alias', 'g':'g_alias'}), self.db['h_alias']]),
				'TechCapConstr': self.db['TechCap'].index}

	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	@property
	def c(self):
		return [{'varName': 'Generation' ,'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['Generation'])},
				{'varName': 'HourlyDemand','value':-self.db['MWP']},
				{'varName': 'Transmission','value':adjMultiIndex.bc(self.db['lineMC'], self.db['h'])},
				{'varName': 'GeneratingCapacity','value': adjMultiIndex.bc(self.db['InvestCost_A'], self.db['id2tech']).droplevel('tech').add(self.db['FOM'],fill_value=0)*1000*len(self.db['h'])/8760}]
	@property
	def u(self):
		return [{'varName': 'HourlyDemand', 'value': self.hourlyLoad},
				{'varName': 'Transmission', 'value': adjMultiIndex.bc(self.db['lineCapacity'], self.db['h'])}]
	@property
	def b_eq(self):
		return [{'constrName': 'equilibrium'}]
	@property
	def A_eq(self):
		return [{'constrName': 'equilibrium', 'varName': 'Generation', 'value'  : appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), ['g','h'],['g_alias2','h_alias'])},
				{'constrName': 'equilibrium', 'varName': 'HourlyDemand', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['HourlyDemand']), ['g','h'],['g_alias2','h_alias'])},
				{'constrName': 'equilibrium', 'varName': 'Transmission', 'value': [appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Transmission']), ['g','h'],['g_alias2','h_alias']),
																				   appIndexWithCopySeries(pd.Series(1-self.db['lineLoss'], index = self.globalDomains['Transmission']), ['g_alias','h'], ['g_alias2','h_alias'])]}]
	@property
	def b_ub(self):
		return [{'constrName': 'ECapConstr'}, {'constrName': 'TechCapConstr', 'value': self.db['TechCap']}]
	@property
	def A_ub(self):
		return [{'constrName': 'ECapConstr', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), ['g','h','id'], ['g_alias','h_alias','id_alias'])},
				{'constrName': 'ECapConstr', 'varName': 'GeneratingCapacity', 'value': -appIndexWithCopySeries(adj.rc_AdjPd(adjMultiIndex.bc(self.hourlyCapFactors, self.db['id2g']), alias = {'h':'h_alias', 'g':'g_alias'}), 'id','id_alias')},
				{'constrName': 'TechCapConstr', 'varName': 'GeneratingCapacity', 'value': adHocMerge(adjMultiIndex.bc(pd.Series(1, index = self.globalDomains['GeneratingCapacity']), self.db['id2tech']), self.db['id2g'])}]

	def postSolve(self, solution, **kwargs):
		if solution['status'] == 0:
			self.unloadToDb(solution)
			self.db['Welfare'] = -solution['fun']
			self.db['FuelConsumption'] = fuelConsumption(self.db)
			self.db['Emissions'] = emissionsFuel(self.db)
			self.db['marginalSystemCosts'] = marginalSystemCosts(self.db)
			self.db['congestionRent'] = congestionRent(self.db)
			self.db['meanConsumerPrice'] = meanMarginalSystemCost(self.db, self.db['HourlyDemand'])


class mEmissionCap(mSimple):
	def __init__(self, db, blocks=None, commonCap = True, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)
		self.commonCap = commonCap

	@property
	def b_ub(self):
		return super().b_ub + [{'constrName': 'emissionsCap', 'value': pyDbs.pdSum(self.db['CO2Cap'], 'g') if self.commonCap else adj.rc_pd(self.db['CO2Cap'], alias = {'g':'g_alias'})}]

	@property
	def A_ub(self):
		return super().A_ub + [{'constrName': 'emissionsCap', 'varName': 'Generation', 'value': adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level = 'EmissionType'), self.globalDomains['Generation']) if self.commonCap else appIndexWithCopySeries(adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation']),'g','g_alias')}]

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
		return super().b_ub + [{'constrName': 'RESCapConstraint', 'value': 0 if self.commonCap else adj.rc_pd(pd.Series(0, index = self.db['RESCap'].index), alias = {'g':'g_alias'})}]

	@property
	def A_ub(self):
		if self.commonCap:
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation', 'value': -1, 'conditions': self.cleanIds},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand', 'value': self.db['RESCap'].mean()}]
		else:
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation']), 'g','g_alias'), 'conditions': self.cleanIds},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand', 'value': appIndexWithCopySeries(adjMultiIndex.bc(self.db['RESCap'], self.globalDomains['HourlyDemand']), 'g', 'g_alias')}]