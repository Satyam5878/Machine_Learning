'''
Created on 03-Mar-2018

@author: i348567
'''
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
import pandas as pd
from math import sqrt,log,exp,pi    
from random import uniform


class Gaussian():
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
    def pdf(self,datum):
        u = (datum-self.mu)/self.sigma
        y = 1/(sqrt(2*pi)*self.sigma)*exp(-u*u/2)
        return y
    def __repr__(self):
        return "Gaussian({0:4.6},{1:4.6})".format(str(self.mu),str(self.sigma))
    
class GaussainMixture():
    def __init__(self, data, mu_min=0,mu_max=0,sigma_min=.1,sigma_max=1,mix=.5):
        self.data = data
        mu_min = min(data)
        mu_max = max(data)
        self.one = Gaussian(uniform(mu_min,mu_max),uniform(sigma_min,sigma_max))
        self.two = Gaussian(uniform(mu_min,mu_max),uniform(sigma_min,sigma_max))
        self.mix = mix
    def pdf(self,datum):
        return self.mix*self.one.pdf(datum)+(1.-self.mix)*self.two.pdf(datum)
    
    def EStep(self):
        print("Here")
        wp1L = []
        wp2L = []
        self.loglike = 0.
        for datum in self.data:
            wp1 = self.one.pdf(datum)*self.mix
            wp2 = self.two.pdf(datum)*(1.-self.mix)
            den = wp1+wp2
            self.loglike += log((wp1+wp2)/den)
            wp1L.append(wp1)
            wp2L.append(wp2)
            #yield (wp1+wp2)
        print(type((wp1L)))
        return ((wp1L,wp2L))
    def MStep(self,weights):
        print("Inside MStep")
        # compute denominators
        print(len(weights))
        (left, rigt) = weights
        one_den = sum(left)
        two_den = sum(rigt)
        # compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(rigt, data))
        # compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(rigt, data)) / two_den)
        # compute new mix
        self.mix = one_den / len(data)
    def __repr__(self):
        return "GaussianMixture({0}, {1}, mix={2.03} )".format(self.one,self.two,str(self.mix))
    def __str__(self):
        return "Mixture: {0}, {1}, mix={2:.03}".format(self.one,self.two,self.mix)
    
    def iterate(self,verbose=False):
       weight = self.EStep()
       self.MStep(weight)
        















## Temporary Data
"""
# generate data
x = np.linspace(start=-10, stop=10, num=1000)

# normalizing the data
y = stats.norm.pdf(x,loc=0,scale=1.5)

# For plotting the gaussian distribution
plt.plot(x,y)
plt.show()
"""
data = pd.read_csv("../../res/Data/bimodal_example.csv")

#print(data.head(10))
""" # for plotting gaussian distribution 
sns.distplot(data, bins=20, kde=False)
plt.show()
"""

""" # for plotting gaussian distribution along with gaussian fitting
sns.distplot(data,bins=20,kde=False,fit=stats.norm)
plt.show()
"""

print(Gaussian(np.mean(data.x),np.std(data.x)))

gaussian_fit = Gaussian(np.mean(data.x),np.std(data.x))


"""
x = np.linspace(-6,8,200)
y = stats.norm(gaussian_fit.mu,gaussian_fit.sigma).pdf(x)
sns.distplot(data, bins=20, kde=False,norm_hist=True)

plt.plot(x,y)
plt.show()

"""


n_iteration = 5
best_mix = None
best_loglike = float('-inf')
mix = GaussainMixture(data.x)
for _ in range(n_iteration):
 #   try:
        mix.iterate(verbose=True)
        if mix.loglike > best_loglike:
            best_loglike = mix.loglike
            best_mix = mix
            
  #  except Exception as e:
  #     print("1 "+str(e))



print(mix.loglike)









