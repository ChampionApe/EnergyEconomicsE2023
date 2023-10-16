from base import *
from lpCompiler import _blocks
from lpModels import modelShell

def adHocMerge(x,y):
	return pd.merge(x.rename('x'), pd.Series(0, index = y).rename('y'), left_index = True, right_index = True).dropna().sum(axis=1)

# A few basic functions for the energy models:
def fuelCost(db):
	return db['FuelPrice'].add(pyDbs.pdSum(db['EmissionIntensity'] * db['EmissionTax'], 'EmissionType'), fill_value=0)

def mc(db):
	return pyDbs.pdSum((db['FuelMix'] * fuelCost(db)).dropna(), 'BFt').add(db['OtherMC'])

def fuelConsumption(db, sumOver='id'):
	return pyDbs.pdSum((db['Generation'] * db['FuelMix']).dropna(), sumOver)

def emissionsFuel(db, sumOver='BFt'):
	return pyDbs.pdSum((db['FuelConsumption'] * db['EmissionIntensity']).dropna(), sumOver)

def plantEmissionIntensity(db):
	return (db['FuelMix'] * db['EmissionIntensity']).groupby('id').sum()

class mSimple(modelShell):
	def __init__(self, db, blocks=None, **kwargs):
		db.updateAlias(alias=[('id','id_constr')])
		super().__init__(db, blocks=blocks, **kwargs)

	def preSolve(self, recomputeMC=False, **kwargs):
		if ('mc' not in self.db.symbols) or recomputeMC:
			self.db['mc'] = mc(self.db)

	@property
	def globalDomains(self):
		return {'Generation': self.db['id'],
				'GeneratingCapacity': self.db['id'],
				'Demand': self.db['c'],
				'ECapConstr': self.db['id_constr'],
				'TechCapConstr': self.db['TechCap'].index}

	@property
	def c(self):
		return [{'varName': 'Generation', 'value': self.db['mc']},
				{'varName': 'Demand', 'value': -self.db['MWP']},
				{'varName': 'GeneratingCapacity','value': adjMultiIndex.bc(self.db['InvestCost_A'], self.db['id2tech']).droplevel('tech').add(self.db['FOM'],fill_value=0)}]
	@property
	def u(self):
		return [{'varName': 'Demand', 'value': self.db['Load']}]

	@property
	def b_eq(self):
		return [{'constrName': 'equilibrium'}]

	@property
	def A_eq(self):
		return [{'constrName': 'equilibrium', 'varName': 'Generation', 'value': 1},
				{'constrName': 'equilibrium', 'varName': 'Demand', 'value': -1}]

	@property
	def b_ub(self):
		return [{'constrName': 'ECapConstr'}, {'constrName': 'TechCapConstr', 'value': self.db['TechCap']}]

	@property
	def A_ub(self):
		return [{'constrName': 'ECapConstr', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), 'id','id_constr')},
				{'constrName': 'ECapConstr', 'varName': 'GeneratingCapacity', 'value': -appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['GeneratingCapacity']), 'id','id_constr')},
				{'constrName': 'TechCapConstr', 'varName': 'GeneratingCapacity', 'value': adjMultiIndex.bc(pd.Series(1, index = self.globalDomains['GeneratingCapacity']), self.db['id2tech'])}]


	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	def postSolve(self, solution, **kwargs):
		if solution['status'] == 0:
			self.unloadToDb(solution)
			self.db['Welfare'] = -solution['fun']
			self.db['FuelConsumption'] = fuelConsumption(self.db)
			self.db['Emissions'] = emissionsFuel(self.db)


class mEmissionCap(mSimple):
	def __init__(self, db, blocks=None, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def b_ub(self):
		return super().b_ub + [{'constrName': 'emissionsCap', 'value': self.db['CO2Cap']}]
	@property
	def A_ub(self):
		return super().A_ub + [{'constrName': 'emissionsCap', 'varName': 'Generation', 'value': plantEmissionIntensity(self.db)}]


class mRES(mSimple):
	def __init__(self, db, blocks=None, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def cleanIds(self):
		s = (self.db['FuelMix'] * self.db['EmissionIntensity']).groupby('id').sum()
		return s[s <= 0].index
	@property
	def b_ub(self):
		return super().b_ub + [{'constrName': 'RESCapConstraint'}]
	@property
	def A_ub(self):
		return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation', 'value': -1, 'conditions': self.cleanIds},
							   {'constrName': 'RESCapConstraint', 'varName': 'Demand', 'value': self.db['RESCap']}]
