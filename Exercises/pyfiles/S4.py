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

class customModel(modelShell):
	def __init__(self, db, blocks = None, **kwargs):
		db.updateAlias(alias = [('h','h_constr'),
								('id','id_constr')])
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
				'HourlyDemand': pd.MultiIndex.from_product([self.db['c'], self.db['h']]),
				'equilibrium': self.db['h_constr']}

	@property
	def c(self):
		return [{'varName': 'Generation', 'value': adjMultiIndex.bc(self.db['mc'], self.db['h'])},
				{'varName': 'HourlyDemand', 'value': -adjMultiIndex.bc(self.db['MWP'], self.globalDomains['HourlyDemand'])}]
	@property
	def u(self):
		return [{'varName': 'Generation', 'value': self.hourlyGeneratingCapacity},
				{'varName': 'HourlyDemand', 'value': self.hourlyLoad_c}]
	@property
	def b_eq(self):
		return [{'constrName': 'equilibrium'}]
	@property
	def A_eq(self):
		return [{'constrName': 'equilibrium', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), 'h','h_constr')},
				{'constrName': 'equilibrium', 'varName': 'HourlyDemand', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['HourlyDemand']), 'h','h_constr')}]

	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	def postSolve(self, solution, **kwargs):
		if solution['status'] == 0:
			self.unloadToDb(solution)
			self.db['Welfare'] = -solution['fun']
			self.db['FuelConsumption'] = fuelConsumption(self.db)
			self.db['Emissions'] = emissionsFuel(self.db)
