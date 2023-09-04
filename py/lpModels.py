from base import *
from scipy import optimize
import itertools
import lpCompiler

def loopxs(x, l, loopName):
    return x.xs(l, level=loopName) if isinstance(x.index, pd.MultiIndex) else x[l]

def updateFromGrids(db, grids, loop, l):
    [db.addOrMerge(g.name, loopxs(g, l, loop.name), priority='second')
     for g in grids]

def readSolutionLoop(sol, loop, i, extract, db):
	return pd.concat(sol[i:len(loop)*len(extract):len(extract)], axis=1).set_axis(loop, axis=1).stack() if isinstance(db[extract[i]], pd.Series) else pd.Series(sol[i:len(loop)*len(extract):len(extract)], index=loop)

class modelShell:
	def __init__(self, db, blocks=None, method = 'highs', scalarDualAtUpper = True, computeDual = True, standardSolve = None, **kwargs):
		self.db = db
		self.method = method
		self.scalarDualAtUpper = True
		self.computeDual = computeDual
		self.blocks = noneInit(blocks, lpCompiler.lpBlock(**kwargs))
		if hasattr(self, 'globalDomains'):
			self.blocks.globalDomains = self.globalDomains

	def __call__(self, execute = None):
		[getattr(self, k)(**v) for k,v in noneInit(execute, dict.fromkeys(['preSolve','initBlocks','solve'], {})).items() if hasattr(self,k)];

	def solve(self, printSol = True, solKwargs = None, solOptions=None, postKwargs = None, **kwargs):
		sol = optimize.linprog(method = self.method, **self.blocks(execute = solKwargs), **noneInit(solOptions, {}))
		if printSol:
			print(f"Solution status {sol['status']}: {sol['message']}")
		self.postSolve(sol, **noneInit(postKwargs, {}))

	def unloadSolution(self, sol):
		fullVector = pd.Series(sol['x'], index=self.blocks.globalVariableIndex)
		return {k: vIndexVariable(fullVector, k, v) for k, v in self.blocks.alldomains.items()}

	def unloadDualSolution(self, sol):
		fullVector = self.blocks.dualSolution(sol, scalarDual = self.scalarDualAtUpper)
		return {**self.unloadShadowValuesConstraints(fullVector), **self.unloadShadowValuesBounds(fullVector)}

	def unloadShadowValuesBounds(self, fullVector):
		return {'λ_'+k: vIndexSymbolDual(fullVector, k, v) for k, v in self.blocks.alldomains.items()}

	def unloadShadowValuesConstraints(self, fullVector):
		return {'λ_'+k: vIndexSymbolDual(fullVector, k, v) for k,v in self.blocks.allconstrdomains.items()}

	def unloadToDb(self, sol):
		[self.db.__setitem__(k, v) for k, v in self.unloadSolution(sol).items()]
		if self.computeDual:
			[self.db.__setitem__(k,v) for k,v in self.unloadDualSolution(sol).items()];

	def loopSolveExtract(self, loop, grids, extract, preSolve=None, initBlocks=None, postSolve=None, printSol=False):
		""" Update exogenous parameters in loop, solve, and extract selected variables """
		n = list(itertools.chain.from_iterable((self.loopSolveExtract_l(loop, grids, extract, l, preSolve = preSolve, initBlocks = initBlocks, postSolve = postSolve, printSol = printSol) for l in loop)))
		return {extract[i]: readSolutionLoop(n, loop, i, extract, self.db) for i in range(len(extract))}
	
	def loopSolveExtract_l(self, loop, grids, extract, l, preSolve=None, initBlocks=None, postSolve=None, printSol=False):
		updateFromGrids(self.db, grids, loop, l)
		if hasattr(self, 'preSolve'):
			self.preSolve(**noneInit(preSolve, {}))
		self.initBlocks(**noneInit(initBlocks, {}))
		self.solve(preSolve=preSolve, initBlocks=initBlocks,
                   postSolve=postSolve, printSol=printSol)
		return [self.db[k] for k in extract]