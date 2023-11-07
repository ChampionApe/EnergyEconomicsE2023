import os
curr = os.getcwd()
_repo = 'EnergyEconomicsE2023'
_repodir = os.path.join(os.getcwd().split(_repo,1)[0],_repo)
_pydir = os.path.join(_repodir,'py')
os.chdir(_pydir)
from base import *
# from baseSparse import *
from lpCompiler import _blocks
from lpModels import modelShell
os.chdir(curr)

# Functions that can be done outside the model:
def fuelCost(db):
	return db['FuelPrice'].add(pyDbs.pdSum(db['EmissionIntensity'] * db['EmissionTax'], 'EmissionType'), fill_value=0)

def mc(db):
	""" Marginal costs in €/GJ """
	return pyDbs.pdSum((db['FuelMix'] * fuelCost(db)).dropna(), 'BFt').add(db['OtherMC'], fill_value=0)

def marginalFuelConsumption(db):
	return (db['Generation']*db['FuelMix']).dropna()

def fuelConsumption(db):
	return pyDbs.pdSum(marginalFuelConsumption(db).dropna(), ['h','id'])

def plantEmissionIntensity(db):
	return pyDbs.pdSum(db['FuelMix'] * db['EmissionIntensity'], 'BFt')

def hourlyEmissions(db):
	return pyDbs.pdSum((db['Generation'] * plantEmissionIntensity(db)),'id')

def Emissions(db):
	return hourlyEmissions(db).sum()

def theoreticalCapacityFactor(db):
	return pyDbs.pdSum( (db['Generation']/pdNonZero(len(db['h']) * db['GeneratingCapacity'])).dropna(), 'h')

def practicalCapacityFactor(model):
	return ( pyDbs.pdSum(model.db['Generation'], 'h')/ pdNonZero(pyDbs.pdSum(model.hourlyGeneratingCap, 'h')) ).dropna()

def marginalSystemCosts(db):
	return adj.rc_pd(db['λ_equilibrium'], alias={'h_constr':'h'}).droplevel('_type')

def meanMarginalSystemCost(db, var):
	tot = pyDbs.pdSum(var, 'h')
	if isinstance(tot,float):
		tot = tot if tot!=0 else np.nan
	else:
		tot[tot==0] = np.nan
	return pyDbs.pdSum( (var * marginalSystemCosts(db)) / tot, 'h')

def capturePrice(db,hvt_type=None):
	if hvt_type is not None:
		return adj.rc_pd((adjMultiIndex.bc(meanMarginalSystemCost(db,db['Generation']), db['id2hvt'])),pd.Index(hvt_type,name='hvt')).dropna().droplevel('hvt')
	else:
		return meanMarginalSystemCost(db,db['Generation'])

def marketValueFactor(db):
	return capturePrice(db,hvt_type=['Wind'])/meanMarginalSystemCost(db, pyDbs.pdSum(db['HourlyDemand'],'c'))

def marginalEconomicValue(model):
	# Multiindices
	mi_power = pd.MultiIndex.from_product([model.db['id'],['Power']],names=['id','CapacityType'])
	mi_energy = pd.MultiIndex.from_product([model.db['id'],['Energy']],names=['id','CapacityType'])
	
	# For standard plants:
	standard_power = adj.rc_pd(pyDbs.pdSum(model.db['λ_Generation'].xs('u',level='_type').abs() * model.hourlyCapFactors, 'h'), getTechs(['Standard'],model.db)).sort_index()
	standard_power.index = adj.rc_pd(mi_power,getTechs(['Standard'],model.db))
	storage_power = adj.rc_pd(pyDbs.pdSum((model.db['λ_discharge'].xs('u',level='_type').abs()+model.db['λ_charge'].xs('u',level='_type').abs()) * model.hourlyCapFactors, 'h'), getTechs(['Storage'],model.db))
	storage_power.index = adj.rc_pd(mi_power,getTechs(['Storage'],model.db))
	storage_energy = adj.rc_pd(pyDbs.pdSum(model.db['λ_stored'].xs('u',level='_type').abs(), 'h'), getTechs(['Storage'],model.db))
	storage_energy.index = adj.rc_pd(mi_energy,getTechs(['Storage'],model.db))
	return pd.concat([standard_power,storage_power,storage_energy],axis=0)

def consumerWelfare(model):
	return ((adjMultiIndex.bc(model.db['MWP'], model.globalDomains['HourlyDemand']) - marginalSystemCosts(model.db)) * model.db['HourlyDemand']).sum()

def producerWelfare(model):
	standard = adj.rc_pd(pyDbs.pdSum(model.db['λ_Generation'].xs('u',level='_type').abs() * model.db['Generation'] , 'h'), getTechs(['Standard'],model.db)) 
	storage = adj.rc_pd(pyDbs.pdSum((model.db['λ_discharge'].xs('u',level='_type').abs() * model.db['discharge'] + model.db['λ_charge'].xs('u',level='_type').abs() * model.db['charge']), 'h'), getTechs(['Storage'],model.db))
	return pd.concat([standard,storage],axis=0)

def getTechs(techs, db):
	""" Subset on tech types"""
	return adj.rc_pd(db['id2tech'], pd.Index(techs if is_iterable(techs) else [techs], name = 'tech')).droplevel('tech')

def subsetIdsTech(x, techs, db):
	return adj.rc_pd(x, getTechs(techs,db))

class mSimple(modelShell):
	""" This class includes 
		(1) An electricity market, 
		(2) Electricity storage """
	def __init__(self, db, blocks = None, **kwargs):
		db.updateAlias(alias=[(k, k+'_constr') for k in ('h','id')])
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def tech(self):
		return ('Standard','Storage')

	@property
	def hourlyCapFactors(self):
		return adjMultiIndex.bc(self.db['CapVariation'], self.db['id2hvt']).droplevel('hvt')
	@property
	def hourlyGeneratingCap(self):
		return subsetIdsTech( (adjMultiIndex.bc(self.db['GeneratingCapacity'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt'),
								('Standard','Storage'), self.db)
	@property
	def hourlyLoad_c(self):
		return self.db['Load'] * self.db['LoadVariation']
	
	@property
	def hourlyLoad(self):
		return pyDbs.pdSum(self.hourlyLoad_c, 'c')

	def preSolve(self, recomputeMC=False, **kwargs):
			if ('mc' not in self.db.symbols) or recomputeMC:
				self.db['mc'] = mc(self.db)
	@property
	def globalDomains(self):
		return {'Generation': adj.rc_pd(pd.MultiIndex.from_product([self.db['h'], self.db['id']]),getTechs(['Standard'],self.db)),
				'discharge' : adj.rc_pd(pd.MultiIndex.from_product([self.db['h'], self.db['id']]),getTechs(['Storage'],self.db)),
				'charge'	: adj.rc_pd(pd.MultiIndex.from_product([self.db['h'], self.db['id']]),getTechs(['Storage'],self.db)),
				'stored'	: adj.rc_pd(pd.MultiIndex.from_product([self.db['h'], self.db['id']]),getTechs(['Storage'],self.db)),			
				'HourlyDemand': pyDbs.cartesianProductIndex([self.db['c'], self.db['h']]),
				'equilibrium': self.db['h_constr'],
				'LawOfMotion': pyDbs.cartesianProductIndex([adj.rc_AdjPd(getTechs('Storage',self.db), alias = {'id':'id_constr'}), self.db['h_constr']])}

	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	@property
	def c(self):
		return [{'varName': 'Generation', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['Generation']),'conditions': getTechs(['Standard'],self.db)},
				{'varName': 'HourlyDemand','value':-adjMultiIndex.bc(self.db['MWP'], self.globalDomains['HourlyDemand'])},
				{'varName': 'discharge', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['discharge']),'conditions': getTechs('Storage',self.db)},
				{'varName': 'charge','value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['charge']),'conditions': getTechs('Storage',self.db)}]

	@property
	def u(self):
		return [{'varName': 'Generation', 'value': adjMultiIndex.bc(self.hourlyGeneratingCap, self.globalDomains['Generation']), 'conditions': getTechs(['Standard'],self.db)},
				{'varName': 'HourlyDemand', 'value': self.hourlyLoad_c},
				{'varName': 'stored', 'value': adjMultiIndex.bc(self.db['sCap'], self.globalDomains['stored'])},
				{'varName': 'discharge', 'value': adjMultiIndex.bc(self.db['GeneratingCapacity'], self.globalDomains['discharge']), 'conditions': getTechs('Storage',self.db)},
				{'varName': 'charge', 'value': adjMultiIndex.bc(self.db['GeneratingCapacity'], self.globalDomains['charge']), 'conditions': getTechs('Storage',self.db)}]
	
	@property
	def b_eq(self):
		return [{'constrName': 'equilibrium'},{'constrName':'LawOfMotion'}]
	
	@property
	def A_eq(self):
		return [{'constrName': 'equilibrium', 'varName': 'Generation', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation']), 'h','h_constr')},
				{'constrName': 'equilibrium', 'varName': 'HourlyDemand', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['HourlyDemand']), 'h','h_constr')},
				{'constrName': 'equilibrium', 'varName': 'discharge', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['discharge']), 'h', 'h_constr')},
				{'constrName': 'equilibrium', 'varName': 'charge', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['charge']), 'h', 'h_constr')},
				{'constrName': 'LawOfMotion', 'varName': 'stored', 'value': [appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['stored']), ['id','h'], ['id_constr','h_constr']),
																				 rollLevelS(appIndexWithCopySeries(adjMultiIndex.bc(-1, self.globalDomains['stored']), ['id','h'], ['id_constr','h_constr']), 'h',1)]},
				{'constrName': 'LawOfMotion', 'varName': 'discharge', 'value': appIndexWithCopySeries(adjMultiIndex.bc(1/self.db['effS'], self.globalDomains['stored']), ['id','h'], ['id_constr','h_constr']), 'conditions': getTechs('Storage',self.db)},
				{'constrName': 'LawOfMotion', 'varName': 'charge', 'value': appIndexWithCopySeries(adjMultiIndex.bc(-self.db['effS'] , self.globalDomains['stored']), ['id','h'], ['id_constr','h_constr']), 'conditions': getTechs('Storage',self.db)}]

	def postSolve(self, solution, **kwargs):
		if solution['status'] == 0:
			self.unloadToDb(solution)
			self.db['Welfare'] = -solution['fun']
			self.db['FuelConsumption'] = fuelConsumption(self.db)
			self.db['hourlyEmissions'] = hourlyEmissions(self.db)
			self.db['Emissions'] = Emissions(self.db)
			self.db['meanConsumerPrice'] = meanMarginalSystemCost(self.db, pyDbs.pdSum(self.db['HourlyDemand'],'c'))
			self.db['capacityFactor'] = practicalCapacityFactor(self)
			self.db['marginalSystemCosts'] = marginalSystemCosts(self.db)
			self.db['marginalEconomicValue'] = marginalEconomicValue(self)
			self.db['capturePrice'] = capturePrice(self.db)
			self.db['marketValueFactor'] = marketValueFactor(self.db)
			self.db['consumerSurplus'] = consumerWelfare(self)
			self.db['producerSurplus'] = producerWelfare(self)
