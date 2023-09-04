from base import *
from baseSparse import *
from scipy.optimize import fsolve, minimize
from scipy import sparse
import numpy as np
import warnings

class UserErrorMessage(UserWarning):
	pass

class modelShellNL:
	def __init__(self, db):
		self.db = db

	def unloadSolutionToDB(self,roots):
		[self.db.__setitem__(i,pd.Series(roots[self.idx_endo[i]],index=self.db[i].index,name=i)) for i in self.idx_endo.keys() if i in self.endo_vars['x_vars']];

	def EquilibriumSolve(self,x0=None,solver='scipy',analyticalJacobian=False,n_iter=None,update_db=True):
		""" Function for finding equilibrium prices using the excess demand function """
		# Check an equilibrium always exists:
		if (pyDbs.pdSum(self.db['LoadVariation'] * self.db['Load'],'c')-pyDbs.pdSum(self.hourlyGeneratingCapacity,'id')>0).any():
		 	warnings.warn(r'You are missing '+str((pyDbs.pdSum(self.db['LoadVariation'] * self.db['Load'],'c')-pyDbs.pdSum(self.hourlyGeneratingCapacity,'id')).round(1).max())+'in generating capacity for an equilibrium to exist', UserErrorMessage)
		# Update parameters:
		self.x = x0.copy()
		n_iter = 100*(len(x0)+1) if n_iter is None else n_iter
		# Now solve for the equilibrium:
		if solver=='scipy':
			fprime = self.Jacobian_of_ExcessDemand if analyticalJacobian else None
			roots = fsolve(lambda x: self.ExcessDemand(x), x0=x0, fprime=fprime, maxfev = n_iter)
		elif solver=='manual':
			roots =  self.manualSolver(x0=x0,n_iter=n_iter)
		else:
			warnings.warn(r'Currently, it is only possible to choose [scipy] or a [manual](ly) defined solver.', UserErrorMessage)
		if np.isclose(self.ExcessDemand(roots),b=0).all():
			self.x = roots
			self.unloadSolutionToDB(roots)
			if update_db:
				self.postSolve()
		else:
			warnings.warn(r'Solver did not find an equilibrium.', UserErrorMessage)

	def sneakySolve(self,x0=None,solver='scipy',analyticalJacobian=True,n_iter=None,sigma_grid=None,update_db=False):
		""" Function for slowly approaching equilibrium using the sigmas as a globalization strategy """
		for sigma in sigma_grid:
			self.sigma = sigma
			self.Solve(x0=x0,solver=solver,analyticalJacobian=analyticalJacobian,n_iter=n_iter,update_db=update_db)
			x0 = self.x.copy()

	# def NestedEstimator(self,x,x0=None,fprime=None,weight_matrix=None):

	# 	while self.

	def Estimate(self,x,x0=None,method='MPEG',fprime=None,weight_matrix=None):
		""" Function for estimating/calibrating model to data """
		# Update xi:
		# old_x = self.idx_x.copy()
		# [self.idx_x.__setitem__(x,theta.xs('level'))
		if weight_matrix is None:
			weight_matrix = np.identity(len(data),dtype=float)
		if method=='MPEG':
			sol = minimize(
				lambda x: self.EstimationObjective(x,weight_matrix), 
				x0=x0,
				constraints = {'type':'eq','fun':self.NonLinearSystem,'jac':self.dNonLinearSystem_dTheta},
				jac = self.fprime
			)
		if method=='Nested':
			sol = minimize(
				lambda x: self.EstimationObjective(),
				x0 = x0,
				jac = self.fprime
			)
		if sol['succes']:
			self.x = sol['x']
			self.unloadSolutionToDB(self.x)
			self.postSolve()
		else:
			warnings.warn(r'Objective function was not minimized.', UserErrorMessage)
		return sol['message']

	def maximizeWelfare(self,x0=None,jacobian=None,constraints=None):
		""" function for solving model as a non-linear programming problem, where economic welfare is maximized """ 
		if jacobian=='analytical':
			jacobian = self.ED_Jacobian
		sol = minimize(
			fun=lambda x: self.Welfare(x),
			x0=x0, 
			constraints = constraints,
			jac=jacobian
		)
		if sol['success']:
			self.x = sol['x']
			self.unloadSolutionToDB(self.x)
			self.postSolve()
		else:
			warnings.warn(r'Economic Welfare was not maximized.', UserErrorMessage)
		return sol['message']

	

	# # Some common model properties irrespective of modelling framework
	# def fuelCost(self,x):
	# 	""" Marginal fuel costs in €/GJ """
	# 	return self.db['FuelPrice'].add(pyDbs.pdSum(self.db['EmissionIntensity'] * self.xArray2pdSeries(x,variable='EmissionTax'), 'EmissionType'), fill_value=0)

	# def mc(self,x):
	# 	""" Marginal costs in €/GJ """
	# 	return pyDbs.pdSum((self.db['FuelMix'] * self.fuelCost(x)).dropna(), 'BFt').add(self.xArray2pdSeries(x,variable='OtherMC'))
	# def fuelConsumption(self,x):
	# 	""" fuel consumption per fuel type """
	# 	return pyDbs.pdSum((self.hourlyGeneration(x) * self.db['FuelMix']).dropna(), ['h','id'])

	# def emissions(self,x):
	# 	""" CO2 emissions """
	# 	return pyDbs.pdSum(self.fuelConsumption(x) * self.db['EmissionIntensity'], 'BFt')