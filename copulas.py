from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class Copula:
    '''
    Parent class for copulas
    '''
    
    # parameters for calculating probabilities numerically
    prob_sample_size = 10000000 # sample size
    prob_band = 0.005 # band size
    prob_sample = np.array([]) # sample     
    
    def log_likelihood(self, u, v):
        '''
        Compute log likelihood for copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: log likelihood (1,)
        '''
        return np.log(self.pdf(u,v)).sum()
    
    def cdf_u_given_v(self, u, v):
        '''
        Compute numberically conditional CDF P(U<=u|V=v)
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (1,)
            v (numpy.ndarray): input (uniform) data with shape (1,)
            
        Returns:
            numpy.ndarray: conditional CDF (1,)
        '''
        # generate sample if it does not exist
        if len(self.prob_sample) == 0:
            self.prob_sample = self.sample(size=self.prob_sample_size)
        # calculate conditional CDF
        s_u = self.prob_sample[:,0]
        s_v = self.prob_sample[:,1]
        condition = (s_v <= v+self.prob_band/2) & (s_v >= v-self.prob_band/2)
        sample_size = len(s_u[condition])
        prob = (s_u[condition] <= u).sum() / sample_size
        return prob
    
    def cdf_v_given_u(self, u, v):
        '''
        Compute numberically conditional CDF P(V<=v|U=u)
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (1,)
            v (numpy.ndarray): input (uniform) data with shape (1,)
            
        Returns:
            numpy.ndarray: conditional CDF (1,)
        '''
        # generate sample if it does not exist
        if len(self.prob_sample) == 0:
            self.prob_sample = self.sample(size=self.prob_sample_size)
        # calculate conditional CDF
        s_u = self.prob_sample[:,0]
        s_v = self.prob_sample[:,1]
        condition = (s_u <= u+self.prob_band/2) & (s_u >= u-self.prob_band/2)
        sample_size = len(s_v[condition])
        prob = (s_v[condition] <= v).sum() / sample_size
        return prob
    
    def plot_sample(self, size=10000):
        '''
        Generate and plot a sample from copula
        
        Args:
            size (int): sample size    
        '''
        
        # generate sample
        sample = self.sample(size=size)
        u = sample[:,0]
        v = sample[:,1]
        # plot
        fig, ax = plt.subplots(figsize=(18,4), nrows=1, ncols=3)
        ax[0].hist(u, density=True)
        ax[0].set(title='Historgram of U')
        ax[1].hist(v, density=True)
        ax[1].set(title='Historgram of V')
        ax[2].scatter(u, v, alpha=0.2)
        ax[2].set(title='Scatterplot of U and V', xlabel='U', ylabel='V')
        
    def plot_with_data(self, u, v):
        '''
        Generate a sample from copula and plot it together with provided (transformed) data u, v
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)   
        '''
        # generate sample
        size = len(x)
        sample = self.sample(size=size)
        s_u = sample[:,0]
        s_v = sample[:,1]
        # plot
        fig, ax = plt.subplots(figsize=(18,4), nrows=1, ncols=3)
        ax[0].scatter(u, v, alpha=0.2, c='tab:orange', label='real')
        ax[0].legend()
        ax[1].scatter(s_u, s_v, alpha=0.2, c='tab:blue', label='generated')
        ax[1].legend()
        ax[2].scatter(u, v, alpha=0.2, c='tab:orange', label='real')
        ax[2].scatter(s_u, s_v, alpha=0.2, c='tab:blue', label='generated')
        ax[2].legend()
        
class GaussianCopula(Copula):
    '''
     Class for bivariate Gaussian copula
    '''
    name = 'Gaussian'
    num_params = 1
    
    def __init__(self, rho=0):
        '''
        Initialize Gaussian copula object
        
        Args:
            rho: copula parameter rho
        '''
        self.rho = rho
        self.cov = np.array([[1,rho],[rho,1]])
        
    def cdf(self, u, v):
        '''
        Compute CDF for Gaussian copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: cumulative probability (n_samples,)
        '''
        data = np.vstack([stats.norm.ppf(u),stats.norm.ppf(v)]).T
        
        return stats.multivariate_normal(cov=self.cov).cdf(data)
    
    def pdf(self, u, v):
        '''
        Compute PDF for Gaussian copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: probability density (n_samples,)
        '''
        #data = np.vstack([[stats.norm.ppf(u),stats.norm.ppf(v)]]).T
        
        #return stats.multivariate_normal(cov=self.cov).pdf(data)
        
        from scipy.special import erfinv
        
        a = np.sqrt(2) * erfinv(2*u - 1)
        b = np.sqrt(2) * erfinv(2*u - 1)
        
        n1 = 1/np.sqrt(1-self.rho**2)
        n2 = np.exp(-((a**2+b**2)*self.rho**2 - 2*a*b*self.rho)/ 2*(1-self.rho**2))
        
        return  n1*n2 
        
    
    def fit(self, u, v):
        '''
        Fit Gaussian copula to data
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
        '''
        tau = stats.kendalltau(u,v)[0]
        self.rho = np.sin(tau * np.pi/2)
        self.cov = np.array([[1,self.rho],[self.rho,1]])
    
    def sample(self, size=1):
        '''
        Generate sample from Gaussian copula
        
        Args:
            size (int): sample size
            
        Returns:
            numpy.ndarray: sample (size,2)
        '''
        smp = stats.multivariate_normal(cov=self.cov).rvs(size=size)
        u = stats.norm.cdf(smp[:,0])
        v = stats.norm.cdf(smp[:,1])
        sample = np.vstack([u,v]).T
        
        return sample
    
class ArchimedeanCopula(Copula):
    '''
    Parent class for Archimedean copulas
    '''
    
    def sample(self, size=1):
        '''
        Generate sample from Archimedean copula using Kendall distribution function
        
        Args:
            size (int): sample size
            
        Returns:
            numpy.ndarray: sample (size,2)
        '''
        # step 1
        t1 = np.random.rand(size)
        t2 = np.random.rand(size)

        # steps 2 and 3
        w = []
        for t in t2:
            func = lambda w: self.K(w) - t
            w.append(brentq(func, 0.0000000001, 0.9999999999))
        w = np.array(w).flatten()

        # step 4
        u = self.phi_inv(t1 * self.phi(w))
        v = self.phi_inv((1-t1) * self.phi(w))
        sample = np.vstack([u,v]).T
        
        return sample
    
class ClaytonCopula(ArchimedeanCopula):
    '''
     Class for bivariate Clayton copula
    '''
    
    name = 'Clayton'
    alpha_bounds = (-1,20)
    num_params = 1
    
    def __init__(self, alpha=-1):
        '''
        Initialize Clayton copula object
        
        Args:
            alpha: copula parameter alpha
        '''
        self.alpha = alpha
        
    def phi(self, t):
        '''
        Compute generator function for Clayton copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of generator function (n_samples,)
        '''
        alpha = self.alpha
        return 1/alpha * (t**(-alpha) - 1)
    
    def phi_inv(self, t):
        '''
        Compute inverse generator function for Clayton copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of inverse generator function (n_samples,)
        '''
        alpha = self.alpha
        return (alpha * t + 1)**(-1/alpha)
    
    def K(self, t):
        '''
        Compute Kendall distribution function for Clayton copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of Kendall distribution function (n_samples,)
        '''
        alpha = self.alpha
        return t * (alpha - t**alpha + 1) / alpha
        
        
    def cdf(self, u, v):
        '''
        Compute CDF for Clayton copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: cumulative probability (n_samples,)
        '''
        alpha=self.alpha
        return (u**(-alpha) + v**(-alpha) - 1)**(-1/alpha)
    
    def pdf(self, u, v):
        '''
        Compute PDF for Clayton copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: probability density (n_samples,)
        '''
        alpha=self.alpha
        num = (u**(-alpha) * v**(-alpha) * (alpha+1) * (-1 + v**(-alpha) + u**(-alpha))**(-1/alpha))
        denom = u * v * (-1 + v**(-alpha) + u**(-alpha))**2
        return num/denom
    
    def fit(self, u, v, method='cml'):
        '''
        Fit Clayton copula to data
        
        Args:
            x (numpy.ndarray): input (uniform) data with shape (n_samples,)
            y (numpy.ndarray): input (uniform) data with shape (n_samples,)
        '''
        if method=='tau':
            tau = stats.kendalltau(u,v)[0]
            self.alpha = 2 * tau / (1 - tau)
        elif method=='cml':
            def fn_to_minimize(alpha, u, v):
                self.alpha = alpha
                return -self.log_likelihood(u,v)
            self.alpha = minimize_scalar(fn_to_minimize, args=(u, v), 
                                         bounds=self.alpha_bounds, method='bounded')['x']

class GumbelCopula(ArchimedeanCopula):
    '''
     Class for bivariate Gumbel copula
    '''
    
    name = 'Gumbel'
    alpha_bounds = (-1,20)
    num_params = 1
    
    def __init__(self, alpha=-1):
        '''
        Initialize Gumbel copula object
        
        Args:
            alpha: copula parameter alpha
        '''
        self.alpha = alpha
        
    def phi(self, t):
        '''
        Compute generator function for Gumbel copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of generator function (n_samples,)
        '''
        alpha = self.alpha
        return (-np.log(t))**alpha
    
    def phi_inv(self, t):
        '''
        Compute inverse generator function for Gumbel copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of inverse generator function (n_samples,)
        '''
        alpha = self.alpha
        return np.exp(-t**(1/alpha))
    
    def K(self, t):
        '''
        Compute Kendall distribution function for Gumbel copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of Kendall distribution function (n_samples,)
        '''
        alpha = self.alpha
        return t * (alpha - np.log(t)) / alpha
        
    def cdf(self, u, v):
        '''
        Compute CDF for Gumbel copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: cumulative probability (n_samples,)
        '''
        alpha=self.alpha
        return np.exp(-((-np.log(u))**alpha + (-np.log(v))**alpha) ** (1/alpha))
    
    def pdf(self, u, v):
        '''
        Compute PDF for Gumbel copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: probability density (n_samples,)
        '''
        alpha=self.alpha
        num1 = (-np.log(u))**alpha * (-np.log(v))**alpha * ((-np.log(u))**alpha + (-np.log(v))**alpha)**(1/alpha)
        num2 = alpha + ((-np.log(u))**alpha + (-np.log(v))**alpha)**(1/alpha) - 1
        num3 = np.exp(-((-np.log(u))**alpha + (-np.log(v))**alpha)**(1/alpha))
        denom = u * v * ((-np.log(u))**alpha + (-np.log(v))**alpha)**2 * np.log(u) * np.log(v)
    
        return (num1 * num2 * num3) / denom
    
    def fit(self, u, v, method='cml'):
        '''
        Fit Gumbel copula to data
        
        Args:
            x (numpy.ndarray): input (uniform) data with shape (n_samples,)
            y (numpy.ndarray): input (uniform) data with shape (n_samples,)
        '''
        if method=='tau':
            tau = stats.kendalltau(u,v)[0]
            self.alpha = 1 / (1 - tau)
        elif method=='cml':
            def fn_to_minimize(alpha, u, v):
                self.alpha = alpha
                return -self.log_likelihood(u,v)
            self.alpha = minimize_scalar(fn_to_minimize, args=(u, v), 
                                         bounds=self.alpha_bounds, method='bounded')['x']
            
class FrankCopula(ArchimedeanCopula):
    '''
     Class for bivariate Frank copula
    '''
    
    name = 'Frank'
    alpha_bounds = (-20,20)
    num_params = 1
    
    def __init__(self, alpha=-1):
        '''
        Initialize Frank copula object
        
        Args:
            alpha: copula parameter alpha
        '''
        self.alpha = alpha
        
    def phi(self, t):
        '''
        Compute generator function for Frank copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of generator function (n_samples,)
        '''
        alpha = self.alpha
        return -np.log((np.exp(-alpha*t) - 1) / (np.exp(-alpha) - 1))
    
    def phi_inv(self, t):
        '''
        Compute inverse generator function for Frank copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of inverse generator function (n_samples,)
        '''
        alpha = self.alpha
        return -1/alpha * np.log((np.exp(-alpha) - 1) / np.exp(t) + 1)
    
    def K(self, t):
        '''
        Compute Kendall distribution function for Frank copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of Kendall distribution function (n_samples,)
        '''
        alpha = self.alpha
        return (t + (1 - np.exp(alpha*t)) * np.log((1-np.exp(alpha*t)) * 
                                               np.exp(-alpha*t+alpha) / (1-np.exp(alpha))) / alpha)
        
    def cdf(self, u, v):
        '''
        Compute CDF for Frank copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: cumulative probability (n_samples,)
        '''
        alpha=self.alpha
        return -1/alpha * np.log(1 + (np.exp(-alpha*u)-1) * (np.exp(-alpha*v)-1) / (np.exp(-alpha)-1))
    
    def pdf(self, u, v):
        '''
        Compute PDF for Frank copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: probability density (n_samples,)
        '''
        alpha=self.alpha
        num = alpha * (np.exp(alpha) - 1) * np.exp(alpha * (u + v + 1))
        denom = (-(np.exp(alpha) - 1) * np.exp(alpha * (u + v)) + 
                 (np.exp(alpha*u)-1) * (np.exp(alpha*v)-1) * np.exp(alpha))**2
        return num/denom
    
    def fit(self, u, v, method='cml'):
        '''
        Fit Frank copula to data
        
        Args:
            x (numpy.ndarray): input (uniform) data with shape (n_samples,)
            y (numpy.ndarray): input (uniform) data with shape (n_samples,)
        '''
        if method=='tau':
            raise NotImplementedError
        elif method=='cml':
            def fn_to_minimize(alpha, u, v):
                self.alpha = alpha
                return -self.log_likelihood(u,v)
            self.alpha = minimize_scalar(fn_to_minimize, args=(u, v), 
                                         bounds=self.alpha_bounds, method='bounded')['x']

class JoeCopula(ArchimedeanCopula):
    '''
     Class for bivariate Joe copula
    '''
    
    name = 'Joe'
    alpha_bounds = (1,20)
    num_params = 1
    
    def __init__(self, alpha=1):
        '''
        Initialize Joe copula object
        
        Args:
            alpha: copula parameter alpha
        '''
        self.alpha = alpha
        
    def phi(self, t):
        '''
        Compute generator function for Joe copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of generator function (n_samples,)
        '''
        alpha = self.alpha
        return -np.log(1 - (1-t)**alpha)
    
    def phi_inv(self, t):
        '''
        Compute inverse generator function for Joe copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of inverse generator function (n_samples,)
        '''
        alpha = self.alpha
        return 1 - (1 - np.exp(-t))**(1/alpha)
    
    def K(self, t):
        '''
        Compute Kendall distribution function for Joe copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of Kendall distribution function (n_samples,)
        '''
        alpha = self.alpha
        return (1/alpha * (1-t)**(-alpha) * (alpha*t*(1-t)**alpha - 
                                             (t-1) * ((1-t)**alpha-1) * np.log(1 - (1-t)**alpha)))
        
    def cdf(self, u, v):
        '''
        Compute CDF for Joe copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: cumulative probability (n_samples,)
        '''
        alpha=self.alpha
        return 1 - ((1-u)**alpha + (1-v)**alpha - (1-u)**alpha * (1-v)**alpha) ** (1/alpha)
    
    def pdf(self, u, v):
        '''
        Compute PDF for Joe copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: probability density (n_samples,)
        '''
        alpha = self.alpha
        
        # fix numerical issues:
        u[u==1] = 0.999999999
        u[u==0] = 0.000000001
        v[v==1] = 0.999999999
        v[v==0] = 0.000000001

        n1 = (1-u)**(alpha-1) * (1-v)**(alpha-1)
        n2 = (-(1-u)**alpha * (1-v)**alpha + (1-u)**alpha + (1-v)**alpha) ** (-2 + 1/alpha)
        n3 = (alpha * ((1-u)**alpha - 1) * ((1-v)**alpha - 1) + alpha*(-(1-u)**alpha * (1-v)**alpha + 
                (1-u)**alpha + (1-v)**alpha) - ((1-u)**alpha-1) * ((1-v)**alpha - 1))
        return n1 * n2 * n3
    
    def fit(self, u, v, method='cml'):
        '''
        Fit Joe copula to data
        
        Args:
            x (numpy.ndarray): input (uniform) data with shape (n_samples,)
            y (numpy.ndarray): input (uniform) data with shape (n_samples,)
        '''
        if method=='tau':
            raise NotImplementedError
        elif method=='cml':
            def fn_to_minimize(alpha, u, v):
                self.alpha = alpha
                return -self.log_likelihood(u,v)
            self.alpha = minimize_scalar(fn_to_minimize, args=(u, v), 
                                         bounds=self.alpha_bounds, method='bounded')['x']

class N13Copula(ArchimedeanCopula):
    '''
     Class for bivariate N13 copula
    '''
    
    name = 'N13'
    alpha_bounds = (0,20)
    num_params = 1
    
    def __init__(self, alpha=1):
        '''
        Initialize N13 copula object
        
        Args:
            alpha: copula parameter alpha
        '''
        self.alpha = alpha
        
    def phi(self, t):
        '''
        Compute generator function for N13 copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of generator function (n_samples,)
        '''
        alpha = self.alpha
        return (1 - np.log(t))**alpha - 1
    
    def phi_inv(self, t):
        '''
        Compute inverse generator function for N13 copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of inverse generator function (n_samples,)
        '''
        alpha = self.alpha
        return np.exp(1 - (t+1)**(1/alpha))
    
    def K(self, t):
        '''
        Compute Kendall distribution function for N13 copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of Kendall distribution function (n_samples,)
        '''
        alpha = self.alpha
        return (t/alpha * (1-np.log(t))**(-alpha) * (alpha * (1-np.log(t))**alpha - 
                                                ((1-np.log(t))**alpha - 1) * (np.log(t) - 1)))
        
    def cdf(self, u, v):
        '''
        Compute CDF for N13 copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: cumulative probability (n_samples,)
        '''
        alpha=self.alpha
        return np.exp(1 - ((1-np.log(u))**alpha + (1-np.log(v))**alpha - 1) ** (1/alpha))
    
    def pdf(self, u, v):
        '''
        Compute PDF for N13 copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: probability density (n_samples,)
        '''
        alpha = self.alpha

        n1 = (1-np.log(u))**alpha * (1-np.log(v))**alpha
        n2 = (alpha + ((1-np.log(u))**alpha + (1-np.log(v))**alpha - 1) ** (1/alpha) - 1)
        n3 = ((1-np.log(u))**alpha + (1-np.log(v))**alpha - 1) ** (1/alpha)
        n4 = np.exp(1 - ((1-np.log(u))**alpha + (1-np.log(v))**alpha - 1) ** (1/alpha))
        d = u * v * (np.log(u)-1) * (np.log(v)-1) * ((1-np.log(u))**alpha + (1-np.log(v))**alpha - 1)**2
        
        return n1*n2*n3*n4 / d
    
    def fit(self, u, v, method='cml'):
        '''
        Fit N13 copula to data
        
        Args:
            x (numpy.ndarray): input (uniform) data with shape (n_samples,)
            y (numpy.ndarray): input (uniform) data with shape (n_samples,)
        '''
        if method=='tau':
            raise NotImplementedError
        elif method=='cml':
            def fn_to_minimize(alpha, u, v):
                self.alpha = alpha
                return -self.log_likelihood(u,v)
            self.alpha = minimize_scalar(fn_to_minimize, args=(u, v), 
                                         bounds=self.alpha_bounds, method='bounded')['x']


class N14Copula(ArchimedeanCopula):
    '''
     Class for bivariate N14 copula
    '''
    
    name = 'N14'
    alpha_bounds = (1,20)
    num_params = 1
    
    def __init__(self, alpha=1):
        '''
        Initialize N14 copula object
        
        Args:
            alpha: copula parameter alpha
        '''
        self.alpha = alpha
        
    def phi(self, t):
        '''
        Compute generator function for N14 copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of generator function (n_samples,)
        '''
        alpha = self.alpha
        return (t**(-1/alpha) - 1)**alpha
    
    def phi_inv(self, t):
        '''
        Compute inverse generator function for N14 copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of inverse generator function (n_samples,)
        '''
        alpha = self.alpha
        return (t**(1/alpha) + 1)**(-alpha)
    
    def K(self, t):
        '''
        Compute Kendall distribution function for N14 copula
        
        Args:
            t (numpy.ndarray): input data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: value of Kendall distribution function (n_samples,)
        '''
        alpha = self.alpha
        return t * (2 - t**(1/alpha))
        
    def cdf(self, u, v):
        '''
        Compute CDF for N14 copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: cumulative probability (n_samples,)
        '''
        alpha=self.alpha
        return (1 + ((u**(-1/alpha)-1)**alpha + (v**(-1/alpha)-1)**alpha)**(1/alpha))**(-alpha)
    
    def pdf(self, u, v):
        '''
        Compute PDF for N14 copula
        
        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)
            
        Returns:
            numpy.ndarray: probability density (n_samples,)
        '''
        alpha = self.alpha
        
        # fix numerical issues:
        #u[u==1] = 0.999999999
        #u[u==0] = 0.000000001
        #v[v==1] = 0.999999999
        #v[v==0] = 0.000000001
        
        a = u**(-1/alpha) * (1 - u**(1/alpha))
        b = v**(-1/alpha) * (1 - v**(1/alpha))
        n1 = (a**alpha * b**alpha * (a**alpha + b**alpha)**((1-2*alpha)/alpha) * 
              ((a**alpha + b**alpha)**(1/alpha) + 1)**(-2-alpha))
        n2 = alpha * (a**alpha + b**alpha)**(1/alpha) + alpha * ((a**alpha + b**alpha)**(1/alpha) + 1) - 1
        d = alpha * u * v * (u**(1/alpha) - 1) * (v**(1/alpha) - 1)
        
        return n1 * n2 / d
        
    
    def fit(self, u, v, method='cml'):
        '''
        Fit N14 copula to data
        
        Args:
            x (numpy.ndarray): input (uniform) data with shape (n_samples,)
            y (numpy.ndarray): input (uniform) data with shape (n_samples,)
        '''
        if method=='tau':
            raise NotImplementedError
        elif method=='cml':
            def fn_to_minimize(alpha, u, v):
                self.alpha = alpha
                return -self.log_likelihood(u,v)
            self.alpha = minimize_scalar(fn_to_minimize, args=(u, v), 
                                         bounds=self.alpha_bounds, method='bounded')['x']
            
