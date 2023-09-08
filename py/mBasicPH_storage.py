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
	return pyDbs.pdSum((db['FuelMix'] * (adjMultiIndex.applyMult(subsetIdsTech(db['Generation_E'], ('Standard (E)','Backpressure'), db), db['g_E2g']).droplevel('g_E').add(
										 adjMultiIndex.applyMult(subsetIdsTech(db['Generation_H'], 'Standard (H)', db), db['g_H2g']).droplevel('g_H'), fill_value = 0))).dropna(), ['h','id'])

def plantEmissionIntensity(db):
	return pyDbs.pdSum(db['FuelMix'] * db['EmissionIntensity'], 'BFt')

def emissionsFuel(db):
	return pyDbs.pdSum(fuelConsumption(db) * db['EmissionIntensity'], 'BFt')

def theoreticalCapacityFactor(db):
	return pyDbs.pdSum((subsetIdsTech(db['Generation_E'], ('Standard (E)','Backpressure'), db) / pdNonZero(len(db['h']) * db['GeneratingCap_E'])).dropna(), 'h').droplevel('g_E')

def marginalSystemCosts(db,market):
	return -adj.rc_AdjPd(db[f'λ_equilibrium_{market}'], alias={'h_constr':'h', f'g_{market}_constr': f'g_{market}'}).droplevel('_type')

def meanMarginalSystemCost(db, var, market):
	return pyDbs.pdSum( (var * marginalSystemCosts(db,market)) / pdNonZero(pyDbs.pdSum(var, 'h')), 'h')

def marginalEconomicValue(m):
	""" Defines over id """
	return pd.Series.combine_first( subsetIdsTech(-pyDbs.pdSum((m.db['λ_Generation_E'].xs('u',level='_type')  * m.hourlyCapFactors).dropna(), 'h').add( 1000 * m.db['FOM'] * len(m.db['h'])/8760, fill_value = 0).droplevel('g_E'),('Standard (E)','Backpressure'), m.db),
									subsetIdsTech(-pyDbs.pdSum((m.db['λ_Generation_H'].xs('u',level='_type')  * m.hourlyCapFactors).dropna(), 'h').add( 1000 * m.db['FOM'] * len(m.db['h'])/8760, fill_value = 0).droplevel('g_H'),('Standard (H)','Heat pump'), m.db)
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
		(4) dynamics, 
		(5) CHP plants and heat pumps """
	def __init__(self, db, blocks = None, **kwargs):
		db.updateAlias(alias=[(k, k+'_constr') for k in ('h','g_E','g_H','g','id')]+[(k, k+'_alias') for k in ['g_E']])
		db['gConnected'] = db['lineCapacity'].index
		db['id2modelTech2tech'] = sortAll(adjMultiIndex.bc(pd.Series(0, index = db['id2tech']), db['tech2modelTech'])).index
		super().__init__(db, blocks=blocks, **kwargs)

	def mapToG(self, symbol, market, alias = None):
		return adjMultiIndex.applyMult(symbol, adj.rc_pd(self.db[f'g_{market}2g'], alias = alias))

	@property
	def modelTech_E(self):
		return ('Standard (E)','Backpressure','Heat pump')
	@property
	def modelTech_H(self):
		return ('Standard (H)','Backpressure','Heat pump')

	@property
	def hourlyCapFactors(self):
		return adjMultiIndex.bc(self.db['CapVariation'], self.db['id2hvt']).droplevel('hvt')
	@property
	def hourlyGeneratingCap_E(self):
		return subsetIdsTech( (adjMultiIndex.bc(self.db['GeneratingCap_E'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt'),
								('Standard (E)','Backpressure'), self.db)
	@property
	def hourlyGeneratingCap_H(self):
		return subsetIdsTech( (adjMultiIndex.bc(self.db['GeneratingCap_H'], self.db['id2hvt']) * self.db['CapVariation']).dropna().droplevel('hvt'),
								('Standard (H)','Heat pump'), self.db)
	@property
	def hourlyLoad_cE(self):
		return adjMultiIndex.bc(self.db['Load_E'] * self.db['LoadVariation_E'], self.db['c_E2g_E']).reorder_levels(['c_E','g_E','h'])
	@property
	def hourlyLoad_cH(self):
		return adjMultiIndex.bc(self.db['Load_H'] * self.db['LoadVariation_H'], self.db['c_H2g_H']).reorder_levels(['c_H','g_H','h'])
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
		return {'Generation_E': pyDbs.cartesianProductIndex([subsetIdsTech(self.db['id2g_E'], self.modelTech_E, self.db), self.db['h']]),
				'Generation_H': pyDbs.cartesianProductIndex([subsetIdsTech(self.db['id2g_H'], self.modelTech_H, self.db), self.db['h']]),
				'discharge_H' : adj.rc_pd(pyDbs.cartesianProductIndex([subsetIdsTech(self.db['id2g_H'], 'HS', self.db), self.db['h']]), self.db['sCap']),
				'charge_H'	: adj.rc_pd(pyDbs.cartesianProductIndex([subsetIdsTech(self.db['id2g_H'], 'HS', self.db), self.db['h']]), self.db['sCap']),
				'stored_H'	: adj.rc_pd(pyDbs.cartesianProductIndex([subsetIdsTech(self.db['id2g_H'], 'HS', self.db), self.db['h']]), self.db['sCap']),				
				'HourlyDemand_E': pyDbs.cartesianProductIndex([self.db['c_E2g_E'], self.db['h']]),
				'HourlyDemand_H': pyDbs.cartesianProductIndex([self.db['c_H2g_H'], self.db['h']]),
				'Transmission_E': pyDbs.cartesianProductIndex([self.db['gConnected'],self.db['h']]),
				'equilibrium_E': pd.MultiIndex.from_product([self.db['g_E_constr'], self.db['h_constr']]),
				'equilibrium_H': pd.MultiIndex.from_product([self.db['g_H_constr'], self.db['h_constr']]),
				'PowerToHeat':	 pyDbs.cartesianProductIndex([adj.rc_AdjPd(getTechs(['Backpressure','Heat pump'],self.db), alias = {'id':'id_constr'}), self.db['h_constr']]),
				'LawOfMotion_H': pyDbs.cartesianProductIndex([adj.rc_AdjPd(getTechs('HS',self.db), alias = {'id':'id_constr'}), self.db['h_constr']])}

	def initBlocks(self, **kwargs):
		[getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

	@property
	def c(self):
		return [{'varName': 'Generation_E', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['Generation_E']), 'conditions': getTechs(['Standard (E)','Backpressure'],self.db)},
				{'varName': 'Generation_H', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['Generation_H']), 'conditions': getTechs(['Standard (H)','Heat pump'],self.db)},
				{'varName': 'HourlyDemand_E','value':-adjMultiIndex.bc(self.db['MWP_E'], self.globalDomains['HourlyDemand_E'])},
				{'varName': 'HourlyDemand_H','value':-adjMultiIndex.bc(self.db['MWP_H'], self.globalDomains['HourlyDemand_H'])},
				{'varName': 'Transmission_E','value': adjMultiIndex.bc(self.db['lineMC'], self.db['h'])},
				{'varName': 'discharge_H', 'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['discharge_H']), 'conditions': getTechs('HS',self.db)},
				{'varName': 'charge_H',	'value': adjMultiIndex.bc(self.db['mc'], self.globalDomains['charge_H']),   'conditions': getTechs('HS',self.db)}]

	@property
	def u(self):
		return [{'varName': 'Generation_E', 'value': adjMultiIndex.bc(self.hourlyGeneratingCap_E, self.globalDomains['Generation_E']), 'conditions': getTechs(['Standard (E)','Backpressure'],self.db)},
				{'varName': 'Generation_H', 'value': adjMultiIndex.bc(self.hourlyGeneratingCap_H, self.globalDomains['Generation_H']), 'conditions': getTechs(['Standard (H)','Heat pump'],self.db)},
				{'varName': 'HourlyDemand_E', 'value': self.hourlyLoad_cE},
				{'varName': 'HourlyDemand_H', 'value': self.hourlyLoad_cH},
				{'varName': 'Transmission_E', 'value': adjMultiIndex.bc(self.db['lineCapacity'], self.db['h'])},
				{'varName': 'stored_H', 'value': adjMultiIndex.bc(self.db['sCap'], self.globalDomains['stored_H'])},
				{'varName': 'discharge_H', 'value': adjMultiIndex.bc(self.db['GeneratingCap_H'], self.globalDomains['discharge_H']), 'conditions': getTechs('HS',self.db)},
				{'varName': 'charge_H', 'value': adjMultiIndex.bc(self.db['chargeCap_H'], self.globalDomains['charge_H']), 'conditions': getTechs('HS',self.db)}]
	@property
	def l(self):
		return [{'varName': 'Generation_E', 'value': -np.inf, 'conditions': getTechs('Heat pump',self.db)}]

	@property
	def b_eq(self):
		return [{'constrName': 'PowerToHeat'}]
	@property
	def A_eq(self):
		return [{'constrName': 'PowerToHeat','varName': 'Generation_E', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Generation_E']), ['id','h'], ['id_constr','h_constr']), 'conditions': getTechs(['Backpressure','Heat pump'],self.db)},
				{'constrName': 'PowerToHeat','varName': 'Generation_H', 'value': appIndexWithCopySeries(adjMultiIndex.bc(-self.db['E2H'], self.globalDomains['Generation_H']), ['id','h'],['id_constr','h_constr']), 'conditions': getTechs(['Backpressure','Heat pump'],self.db)}]
	@property
	def b_ub(self):
		return [{'constrName': 'equilibrium_E'}, {'constrName': 'equilibrium_H'}, {'constrName': 'LawOfMotion_H'}]
	@property
	def A_ub(self):
		return [{'constrName': 'equilibrium_E', 'varName': 'Generation_E', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation_E']), ['g_E','h'],['g_E_constr','h_constr'])},
				{'constrName': 'equilibrium_E', 'varName': 'HourlyDemand_E', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['HourlyDemand_E']), ['g_E','h'],['g_E_constr','h_constr'])},
				{'constrName': 'equilibrium_E', 'varName': 'Transmission_E', 'value': [appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['Transmission_E']), ['g_E','h'],['g_E_constr','h_constr']),
																					   appIndexWithCopySeries(pd.Series(self.db['lineLoss']-1, index = self.globalDomains['Transmission_E']), ['g_E_alias','h'], ['g_E_constr','h_constr'])]},
				{'constrName': 'equilibrium_H', 'varName': 'Generation_H', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['Generation_H']), ['g_H','h'],['g_H_constr','h_constr'])},
				{'constrName': 'equilibrium_H', 'varName': 'HourlyDemand_H','value':appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['HourlyDemand_H']), ['g_H','h'],['g_H_constr','h_constr'])},
				{'constrName': 'equilibrium_H', 'varName': 'discharge_H', 'value': appIndexWithCopySeries(pd.Series(-1, index = self.globalDomains['discharge_H']), ['g_H','h'], ['g_H_constr','h_constr'])},
				{'constrName': 'equilibrium_H', 'varName': 'charge_H', 'value': appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['charge_H']), ['g_H','h'], ['g_H_constr','h_constr'])},
				{'constrName': 'LawOfMotion_H', 'varName': 'stored_H', 'value': [appIndexWithCopySeries(pd.Series(1, index = self.globalDomains['stored_H']), ['id','h'], ['id_constr','h_constr']),
																				 rollLevelS(appIndexWithCopySeries(adjMultiIndex.bc(self.db['selfDischarge']-1, self.globalDomains['stored_H']), ['id','h'], ['id_constr','h_constr']), 'h',1)]},
				{'constrName': 'LawOfMotion_H', 'varName': 'discharge_H', 'value': appIndexWithCopySeries(adjMultiIndex.bc(1/self.db['effD'], self.globalDomains['stored_H']), ['id','h'], ['id_constr','h_constr']), 'conditions': getTechs('HS',self.db)},
				{'constrName': 'LawOfMotion_H', 'varName': 'charge_H', 'value': appIndexWithCopySeries(adjMultiIndex.bc(-self.db['effC'] , self.globalDomains['stored_H']), ['id','h'], ['id_constr','h_constr']), 'conditions': getTechs('HS',self.db)}]

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
		return super().b_ub + [{'constrName': 'emissionsCap', 'value': pyDbs.pdSum(self.db['CO2Cap'],'g') if self.commonCap else adj.rc_pd(self.db['CO2Cap'], alias = {'g': 'g_constr'})}]

	@property
	def A_ub(self):
		if self.commonCap:
			return super().A_ub + [{'constrName': 'emissionsCap', 'varName': 'Generation_E', 'value': adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_E']), 'conditions': getTechs(['Standard (E)','Backpressure'],self.db)},
								   {'constrName': 'emissionsCap', 'varName': 'Generation_H', 'value': adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_H']), 'conditions': getTechs('Standard (H)',self.db)}]
		else:
			return super().A_ub + [{'constrName': 'emissionsCap', 'varName': 'Generation_E', 'value': self.mapToG(adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_E']), 'E', alias = {'g':'g_constr'}), 'conditions': getTechs(['Standard (E)', 'Backpressure'], self.db)},
								   {'constrName': 'emissionsCap', 'varName': 'Generation_H', 'value': self.mapToG(adjMultiIndex.bc(plantEmissionIntensity(self.db).xs('CO2',level='EmissionType'), self.globalDomains['Generation_H']), 'H', alias = {'g':'g_constr'}), 'conditions': getTechs('Standard (H)', self.db)}]

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
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation_E', 'value': -1, 'conditions': ('and', [self.cleanIds, getTechs(['Standard (E)','Backpressure'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'Generation_H', 'value': -1, 'conditions': ('and', [self.cleanIds, getTechs(['Standard (H)','Heat pump'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_E','value': self.db['RESCap'].mean()},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_H','value': self.db['RESCap'].mean()}]
		else:
			return super().A_ub + [{'constrName': 'RESCapConstraint', 'varName': 'Generation_E',  'value': self.mapToG(pd.Series(-1, index = self.globalDomains['Generation_E']), 'E', alias = {'g':'g_constr'}), 'conditions': ('and', [self.cleanIds, getTechs(['Standard (E)','Backpressure'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'Generation_H',  'value': self.mapToG(pd.Series(-1, index = self.globalDomains['Generation_H']), 'H', alias = {'g':'g_constr'}), 'conditions': ('and', [self.cleanIds, getTechs(['Standard (H)','Heat pump'],self.db)])},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_E','value': adj.rc_pd(self.mapToG(pd.Series(1, index = self.globalDomains['HourlyDemand_E']), 'E') * self.db['RESCap'], alias = {'g':'g_constr'})},
								   {'constrName': 'RESCapConstraint', 'varName': 'HourlyDemand_H','value': adj.rc_pd(self.mapToG(pd.Series(1, index = self.globalDomains['HourlyDemand_H']), 'H') * self.db['RESCap'], alias = {'g':'g_constr'})}]