from base import *
from lpCompiler import _blocks
from lpModels import modelShell

# Functions for all technology types:
def fuelCost(db):
	return db['FuelPrice'].add(pyDbs.pdSum(db['EmissionIntensity'] * db['EmissionTax'], 'EmissionType'), fill_value=0)

def mc(db):
	""" Marginal costs in €/GJ """
	return pyDbs.pdSum((db['FuelMix'] * fuelCost(db)).dropna(), 'BFt').add(db['OtherMC'], fill_value=0)

def fuelConsumption(db):
	return pyDbs.pdSum((db['FuelMix'] * (subsetIdsTech(db['Generation_E'], ('standard_E','BP'), db).add(
								  subsetIdsTech(db['Generation_H'], 'standard_H', db), fill_value = 0))).dropna(), ['h','id'])

def plantEmissionIntensity(db):
	return pyDbs.pdSum(db['FuelMix'] * db['EmissionIntensity'], 'BFt')

def emissionsFuel(db):
	return pyDbs.pdSum(fuelConsumption(db) * db['EmissionIntensity'], 'BFt')

def theoreticalCapacityFactor(db):
	return pyDbs.pdSum((subsetIdsTech(db['Generation_E'], ('standard_E','BP'), db) / pdNonZero(len(db['h']) * db['GeneratingCap_E'])).dropna(), 'h').droplevel('g')

def marginalSystemCosts(db,market):
	return -adj.rc_AdjPd(db[f'λ_equilibrium_{market}'], alias={'h_alias':'h', 'g_alias2': 'g'}).droplevel('_type')

def meanMarginalSystemCost(db, var, market):
	return pyDbs.pdSum( (var * marginalSystemCosts(db,market)) / pdNonZero(pyDbs.pdSum(var, 'h')), 'h')

def marginalEconomicValue(m):
	""" Defines over id """
	return pd.Series.combine_first( subsetIdsTech(-pyDbs.pdSum((m.db['λ_Generation_E'].xs('u',level='_type')  * m.hourlyCapFactors).dropna(), 'h').add( 1000 * m.db['FOM'] * len(m.db['h'])/8760, fill_value = 0).droplevel('g'),('standard_E','BP'), m.db),
									subsetIdsTech(-pyDbs.pdSum((m.db['λ_Generation_H'].xs('u',level='_type')  * m.hourlyCapFactors).dropna(), 'h').add( 1000 * m.db['FOM'] * len(m.db['h'])/8760, fill_value = 0).droplevel('g'),('standard_H','HP'), m.db)
									)

def getTechs(techs, db):
	""" Subset on tech types"""
	return adj.rc_pd(db['id2modelTech2tech'].droplevel('tech'), pd.Index(techs if is_iterable(techs) else [techs], name = 'modelTech')).droplevel('modelTech')

def getTechs_i(techs, db):
	""" Subset on tech types"""
	return adj.rc_pd(db['id2modelTech2tech'].droplevel('modelTech'), pd.Index(techs if is_iterable(techs) else [techs], name = 'tech')).droplevel('tech')

def subsetIdsTech(x, techs, db):
	return adj.rc_pd(x, getTechs(techs,db))

def subsetIdsTech_i(x, techs, db):
	return adj.rc_pd(x, getTechs_i(techs,db))

class mSimple(modelShell):
	""" This class includes 
		(1) Electricity and heat markets, 
		(2) multiple geographic areas, 
		(3) trade in electricity, 
s		(4) intermittency in generation, 
		(5) CHP plants and heat pumps """
	def __init__(self, db, blocks = None, **kwargs):
		db.updateAlias(alias=[('h','h_alias'), ('g','g_alias'),('g','g_alias2'),('id','id_alias')])
		db['gConnected'] = db['lineCapacity'].index
		db['id2modelTech2tech'] = sortAll(adjMultiIndex.bc(pd.Series(0, index = db['id2tech']), db['tech2modelTech'])).index
		super().__init__(db, blocks=blocks, **kwargs)

	@property
	def modelTech_E(self):
		return ('standard_E','BP','HP')
	@property
	def modelTech_H(self):
		return ('standard_H','BP','HP')
	@property
	def hourlyCapFactors(self):
		return adjMultiIndex.bc(self.db['CapVariation'], self.db['id2hvt']).droplevel('hvt')
	@property
	def hourlyGeneratingCap_E(self):
		return subsetIdsTech( (adjMultiIndex.bc(self.db['GeneratingCap_E'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt'),
								('standard_E','BP'), self.db)
	@property
	def hourlyGeneratingCap_H(self):
		return subsetIdsTech( (adjMultiIndex.bc(self.db['GeneratingCap_H'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt'),
								('standard_H','HP'), self.db)
	@property
	def hourlyLoad_cE(self):
		return adjMultiIndex.bc(self.db['Load_E'] * self.db['LoadVariation_E'], self.db['c_E2g']).reorder_levels(['c_E','g','h'])
	@property
	def hourlyLoad_cH(self):
		return adjMultiIndex.bc(self.db['Load_H'] * self.db['LoadVariation_H'], self.db['c_H2g']).reorder_levels(['c_H','g','h'])
	@property
	def hourlyLoad_E(self):
		return pyDbs.pdSum(self.hourlyLoad_cE, 'c_E')
	@property
	def hourlyLoad_H(self):
		return pyDbs.pdSum(self.hourlyLoad_cH, 'c_H')

	def preSolve(self, recomputeMC=False, **kwargs):
			if ('mc' not in self.db.symbols) or recomputeMC:
				self.db['mc'] = mc(self.db)

	@property
	def globalDomains(self):
		return {'Generation_E': pyDbs.cartesianProductIndex([subsetIdsTech(self.db['id2g'], self.modelTech_E, self.db), self.db['h']]),
				'Generation_H': pyDbs.cartesianProductIndex([subsetIdsTech(self.db['id2g'], self.modelTech_H, self.db), self.db['h']]),
				'HourlyDemand_E': pyDbs.cartesianProductIndex([self.db['c_E2g'], self.db['h']]),
				'HourlyDemand_H': pyDbs.cartesianProductIndex([self.db['c_H2g'], self.db['h']]),
				'Transmission_E': pyDbs.cartesianProductIndex([self.db['gConnected'],self.db['h']]),
				'equilibrium_E': pd.MultiIndex.from_product([self.db['g_alias2'], self.db['h_alias']]),
				'equilibrium_H': pd.MultiIndex.from_product([self.db['g_alias2'], self.db['h_alias']]),
				'PowerToHeat': pyDbs.cartesianProductIndex([adj.rc_AdjPd(getTechs(['BP','HP'],self.db), alias = {'id':'id_alias'}), self.db['h_alias']])}

	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	@property
	def c(self):
		return [{'varName': 'Generation_E', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['Generation_E']), 'conditions': getTechs(['standard_E','BP'],self.db)},
				{'varName': 'Generation_H', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['Generation_H']), 'conditions': getTechs(['standard_H','HP'],self.db)},
				{'varName': 'HourlyDemand_E','value':-adjMultiIndex.bc(self.db['MWP_E'], self.globalDomains['HourlyDemand_E'])},
				{'varName': 'HourlyDemand_H','value':-adjMultiIndex.bc(self.db['MWP_H'], self.globalDomains['HourlyDemand_H'])},
				{'varName': 'Transmission_E','value': adjMultiIndex.bc(self.db['lineMC'], self.db['h'])}]
	@property
	def u(self):
		return [{'varName': 'Generation_E', 'value': adjMultiIndex.bc(self.hourlyGeneratingCap_E, self.globalDomains['Generation_E']), 'conditions': getTechs(['standard_E','BP'],self.db)},
				{'varName': 'Generation_H', 'value': adjMultiIndex.bc(self.hourlyGeneratingCap_H, self.globalDomains['Generation_H']), 'conditions': getTechs(['standard_H','HP'],self.db)},
				{'varName': 'HourlyDemand_E','value':self.hourlyLoad_cE},
				{'varName': 'HourlyDemand_H','value':self.hourlyLoad_cH},
				{'varName': 'Transmission_E', 'value': adjMultiIndex.bc(self.db['lineCapacity'], self.db['h'])}]
	@property
	def l(self):
		return [{'varName': 'Generation_E', 'value': -np.inf, 'conditions': getTechs('HP',self.db)}]
	@property
	def b_eq(self):
		return [{'constrName': 'PowerToHeat'}]
	@property
	def A_eq(self):
		return [{'constrName': 'PowerToHeat', 'varName': 'Generation_E', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation_E']), ['id','h'], ['id_alias','h_alias']), 'conditions': getTechs(['BP','HP'],self.db)},
				{'constrName': 'PowerToHeat', 'varName': 'Generation_H', 'value': appIndexWithCopySeries(adjMultiIndex.bc(-self.db['E2H'], self.globalDomains['Generation_H']), ['id','h'],['id_alias','h_alias']), 'conditions': getTechs(['BP','HP'],self.db)}]
	@property
	def b_ub(self):
		return [{'constrName': 'equilibrium_E'}, {'constrName': 'equilibrium_H'}]
	@property
	def A_ub(self):
		return [{'constrName': 'equilibrium_E', 'varName': 'Generation_E', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation_E']), ['g','h'],['g_alias2','h_alias'])},
				{'constrName': 'equilibrium_E', 'varName': 'HourlyDemand_E','value':appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['HourlyDemand_E']), ['g','h'],['g_alias2','h_alias'])},
				{'constrName': 'equilibrium_E', 'varName': 'Transmission_E','value': [appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Transmission_E']), ['g','h'],['g_alias2','h_alias']),
																					  appIndexWithCopySeries(pd.Series(self.db['lineLoss']-1, index = self.globalDomains['Transmission_E']), ['g_alias','h'], ['g_alias2','h_alias'])]},
				{'constrName': 'equilibrium_H', 'varName': 'Generation_H', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation_H']), ['g','h'],['g_alias2','h_alias'])},
				{'constrName': 'equilibrium_H', 'varName': 'HourlyDemand_H','value':appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['HourlyDemand_H']), ['g','h'],['g_alias2','h_alias'])}]

	def postSolve(self, solution, **kwargs):
		if solution['status'] == 0:
			self.unloadToDb(solution)
			self.db['Welfare'] = -solution['fun']
			self.db['FuelConsumption'] = fuelConsumption(self.db)
			self.db['Emissions'] = emissionsFuel(self.db)
			self.db['marginalSystemCosts_E'] = marginalSystemCosts(self.db, 'E')
			self.db['marginalSystemCosts_H'] = marginalSystemCosts(self.db, 'H')
			self.db['marginalEconomicValue'] = marginalEconomicValue(self)
			self.db['meanConsumerPrice_E'] = meanMarginalSystemCost(self.db, self.db['HourlyDemand_E'],'E')
			self.db['meanConsumerPrice_H'] = meanMarginalSystemCost(self.db, self.db['HourlyDemand_H'],'H')

class mEmissionCap(mSimple):
	def __init__(self, db, blocks = None, commonCap = True, **kwargs):
		super().__init__(db, blocks=blocks, **kwargs)
		self.commonCap = commonCap

	@property
	def b_ub(self):
		return super().b_ub + [{'constrName': 'emissionsCap', 'value': pyDbs.pdSum(self.db['CO2Cap'],'g') if self.commonCap else adj.rc_pd(self.db['CO2Cap'], alias = {'g': 'g_alias'})}]

	@property
	def A_ub(self):
		if self.commonCap:
			return super().A_ub + [{'constrName': 'emissionsCap', 'varName': 'Generation_E', 'value': adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_E']), 'conditions': getTechs(['standard_E','BP'],self.db)},
								   {'constrName': 'emissionsCap', 'varName': 'Generation_H', 'value': adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_H']), 'conditions': getTechs('standard_H',self.db)}]
		else:
			return super().A_ub + [{'constrName': 'emissionsCap', 'varName': 'Generation_E', 'value': appIndexWithCopySeries(adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_E']),'g','g_alias'), 'conditions': getTechs(['standard_E','BP'],self.db)},
								   {'constrName': 'emissionsCap', 'varName': 'Generation_H', 'value': appIndexWithCopySeries(adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_H']),'g','g_alias'), 'conditions': getTechs('standard_H',self.db)}]

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
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation_E', 'value': -1, 'conditions': ('and', [self.cleanIds, getTechs(['standard_E','BP'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'Generation_H', 'value': -1, 'conditions': ('and', [self.cleanIds, getTechs(['standard_H','HP'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_E','value': self.db['RESCap'].mean()},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_H','value': self.db['RESCap'].mean()}]
		else:
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation_E', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation_E']), 'g','g_alias'), 'conditions': ('and', [self.cleanIds, getTechs(['standard_E','BP'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'Generation_H', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation_H']), 'g','g_alias'), 'conditions': ('and', [self.cleanIds, getTechs(['standard_H','HP'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_E', 'value': appIndexWithCopySeries(adjMultiIndex.bc(self.db['RESCap'], self.globalDomains['HourlyDemand_E']), 'g', 'g_alias')},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_H', 'value': appIndexWithCopySeries(adjMultiIndex.bc(self.db['RESCap'], self.globalDomains['HourlyDemand_H']), 'g', 'g_alias')}]