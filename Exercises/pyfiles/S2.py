import numpy as np, pandas as pd
from scipy import optimize, stats

class MAC:
	def __init__(self, α = .5, γ = 1, pe = 1, ϕ = .25, γd = 100):
		""" Initialize with default values in key-value database. """
		self.db = {'α': α, 'γ': γ, 'pe': pe, 'ϕ': ϕ, 'γd': γd}
		
	def C(self, E, **kwargs):
		return self.db['γ'] * np.power(E, self.db['α'])-self.db['pe']*E
	
	def M(self, E, **kwargs):
		return self.db['ϕ']*E
	
	def Ctilde(self, E, **kwargs):
		return self.C(E)-self.db['γd']*np.power(self.M(E),2)/2
	
	def MAC(self, E):
		return (self.db['α']*self.db['γ']*np.power(E, self.db['α']-1)-self.db['pe'])/self.db['ϕ'] 
	
	def E0(self):
		return (self.db['γ']*self.db['α']/self.db['pe'])**(1/(1-self.db['α']))
	
	def C0(self):
		return self.db['γ']*self.E0()**self.db['α']-self.db['pe']*self.E0()
	
	def M0(self):
		return self.db['ϕ']*self.E0()

	def baseSol(self):
		return {'E0': self.E0(), 'C0': self.C0(), 'M0': self.M0()}
	
	def Eopt(self, x0 = 0.5):
		return optimize.fsolve(lambda E: self.db['α']*self.db['γ']*E**(self.db['α']-1)-self.db['pe']-self.db['ϕ']*self.db['γd']*self.M(E), 
							   x0 = x0)   

	def optSol(self, x0 = 0.5):
		Eopt = self.Eopt(x0 = x0)
		return {'Ept': Eopt, 'C': self.Ctilde(Eopt), 'M': self.M(Eopt)}

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
			self.db['σ'] = pd.Series(1, index = self.db['Tech'], name = 'σ')

	def aOpt_i(self, md, i):
		return stats.norm.cdf(
			(np.log(md/self.db['c'][i])+self.db['σ'][i]**2/2)/self.db['σ'][i]
		)
	def fOpt_i(self, md, i):
		return self.db['c'][i] * stats.norm.cdf(
			(np.log(md/self.db['c'][i])-self.db['σ'][i]**2/2)/self.db['σ'][i]
		)
	
	def aOpt_sum(self, md):
		return sum(self.db['θ'] * stats.norm.cdf(
			(np.log(md/self.db['c'])+self.db['σ']**2/2)/self.db['σ']
		))
	
	def fOpt_sum(self, md):
		return sum(self.db['θ'] * self.db['c'] * stats.norm.cdf(
			(np.log(md/self.db['c'])-self.db['σ']**2/2)/self.db['σ']
		))

	def C(self, E, M, **kwargs):
		return super().C(E)-M*self.fOpt_sum(self.db['γd'] * M)
	
	def Ctilde(self, E, M, **kwargs):
		return self.C(E,M)-self.db['γd']*M**2/2

	def FOC_RHS(self, md):
		""" Right-side of equation (7) in lecture note"""
		return md * (1-self.aOpt_sum(md)) + self.fOpt_sum(md)

	def Mopt(self, x0 = None):
		return optimize.fsolve(lambda M: self.db['α'] * self.db['γ'] * (M/(self.db['ϕ']*(1-self.aOpt_sum(self.db['γd'] * M))))**(self.db['α']-1)-self.db['pe']-self.db['ϕ']*self.FOC_RHS(self.db['γd'] * M),
							   x0 = self.M0()/2 if x0 is None else x0)
	
	def Eopt(self, x0 = None):
		Mopt = self.Mopt(x0 = x0)
		return Mopt/(self.db['ϕ']*(1-self.aOpt_sum(self.db['γd'] * Mopt)))

	def aOpt_sum_vec(self, md):
		""" Similar to self.aOpt_sum, but with md being a vector"""
		vecToMat = pd.DataFrame(np.tile(md, (len(self.db['θ']),1)).T, columns = self.db['Tech']) # repeat vector to a matrix with column index 'Tech'
		return (self.db['θ'] * ((np.log(vecToMat/self.db['c'])+self.db['σ']**2/2)/self.db['σ']).apply(stats.norm.cdf)).sum(axis=1)

	def fOpt_sum_vec(self, md):
		""" Similar to self.fOpt_sum, but with md being a vector"""
		vecToMat = pd.DataFrame(np.tile(md, (len(self.db['θ']),1)).T, columns = self.db['Tech']) # repeat vector to a matrix with column index 'Tech'
		return (self.db['θ'] * self.db['c'] * ((np.log(vecToMat/self.db['c'])-self.db['σ']**2/2)/self.db['σ']).apply(stats.norm.cdf)).sum(axis=1)

	def FOC_RHS_vec(self, md):
		""" Right-side of equation (7) in lecture note"""
		return md * (1-self.aOpt_sum_vec(md)) + self.fOpt_sum_vec(md)
