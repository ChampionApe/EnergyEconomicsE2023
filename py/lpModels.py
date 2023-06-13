from base import *
from scipy import optimize
import lpCompiler

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
