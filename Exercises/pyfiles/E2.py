import numpy as np, pandas as pd
from scipy import optimize, stats

class MAC:
	def __init__(self):
		pass



class MACTech(MAC):
	def __init__(self, α = .5, γ = 1, pe = 1, ϕ = .25, γd = 100, θ = None, c = None, σ = None):
		super().__init__(α = α, γ = γ, pe = pe, ϕ = ϕ, γd = γd) # use __init__ method from parent class
		self.initTechs(θ = θ, c = c, σ = σ)
		
	def initTechs(self, θ = None, c = None, σ = None):
		""" Initialize technologies from default values """
		if θ is None:
			self.db['Tech'] = pd.Index(['T1'], name = 'i')
			self.db['θ'] = pd.Series(0, index = self.db['Tech'], name = 'θ')
			self.db['c'] = pd.Series(1, index = self.db['Tech'], name = 'c')
			self.db['σ'] = pd.Series(1, index = self.db['Tech'], name = 'σ')
		elif isinstance(θ, pd.Series):
			self.db['θ'] = θ
			self.db['c'] = c
			self.db['σ'] = σ
			self.db['Tech'] = self.db['θ'].index
		else:
			self.db['Tech'] = 'T'+pd.Index(range(1, len(θ)+1), name = 'i').astype(str)
			self.db['θ'] = pd.Series(θ, index = self.db['Tech'], name = 'θ')
			self.db['c'] = pd.Series(c, index = self.db['Tech'], name = 'c')
			self.db['σ'] = pd.Series(σ, index = self.db['Tech'], name = 'σ')
