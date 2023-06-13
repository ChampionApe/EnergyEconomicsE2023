from base import *
from lpCompiler import _blocks
from lpModels import modelShell

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
        super().__init__(db, blocks=blocks, **kwargs)

    def preSolve(self, recomputeMC=False, **kwargs):
        if ('mc' not in self.db.symbols) or recomputeMC:
            self.db['mc'] = mc(self.db)

    @property
    def globalDomains(self):
        return {'Generation': self.db['id']}

    @property
    def getLoad(self):
        return sum(self.db['Load']) if is_iterable(self.db['Load']) else self.db['Load']

    @property
    def c(self):
        return [{'varName': 'Generation', 'value': self.db['mc']}]
    @property
    def u(self):
        return [{'varName': 'Generation', 'value': self.db['GeneratingCapacity']}]
    @property
    def b_eq(self):
        return [{'constrName': 'equilibrium', 'value': self.getLoad}]
    @property
    def A_eq(self):
        return [{'constrName': 'equilibrium', 'varName': 'Generation', 'value': 1}]

    def initBlocks(self, **kwargs):
        [getattr(self.blocks, f'add_{t}')(**v) for t in _blocks if hasattr(self,t) for v in getattr(self,t)];

    def postSolve(self, solution, **kwargs):
        if solution['status'] == 0:
            self.unloadToDb(solution)
            self.db['SystemCosts'] = solution['fun']
            self.db['FuelConsumption'] = fuelConsumption(self.db)
            self.db['Emissions'] = emissionsFuel(self.db)


class mEmissionCap(mSimple):
    def __init__(self, db, blocks=None, **kwargs):
        super().__init__(db, blocks=blocks, **kwargs)

    @property
    def b_ub(self):
        return [{'constrName': 'emissionsCap', 'value': self.db['CO2Cap']}]
    @property
    def A_ub(self):
        return [{'constrName': 'emissionsCap', 'varName': 'Generation', 'value': plantEmissionIntensity(self.db)}]


class mRES(mSimple):
    def __init__(self, db, blocks=None, **kwargs):
        super().__init__(db, blocks=blocks, **kwargs)

    @property
    def cleanIds(self):
        s = (self.db['FuelMix'] * self.db['EmissionIntensity']).groupby('id').sum()
        return s[s <= 0].index
    @property
    def b_ub(self):
        return [{'constrName': 'RESCapConstraint', 'value': -self.db['RESCap']*self.getLoad}]
    @property
    def A_ub(self):
        return [{'constrName': 'RESCapConstraint', 'varName': 'Generation', 'value': -1, 'conditions': self.cleanIds}]
