import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats


#Perform Gibbs sampling on a 1D Gaussian mixture model

class GMM:
    def __init__(self, k, mu, nu, pi) -> None:
        self.k = k
        self.mu = mu
        self.nu = nu
        self.pi = pi
        
    def sample(self, n):
        samples = []
        Z = []
        for i in range(n):
            z = np.random.choice(self.k, p=self.pi)
            Z.append(z)
            samples.append(np.random.normal(self.mu[z], 1/np.sqrt(self.nu[z])))
            
        return samples, Z
    
    def print_parameters(self):
        print("mu: ", self.mu)
        print("nu: ", self.nu)
        print("pi: ", self.pi)
        

def Beta(a):
    #a is a vector of length k
    num = 1
    for i in range(len(a)):
        num *= math.gamma(a[i])
    
    den = math.gamma(sum(a))
    
    return num/den
        

class CollapsedGibbs:
    
    def __init__(self, X, k, m, p, a, b) -> None:
        self.X = np.array(X)
        self.N = len(X)
        self.m = m
        self.p = p
        self.a = a
        self.b = b
        self.K = k
        self.pi = np.array([1/k for i in range(k)], dtype=float)
        self.beta = np.array([1.0 for i in range(k)], dtype=float)
        self.mu = np.array([0.0 for i in range(k)], dtype=float)
        self.nu = np.array([10.0 for i in range(k)], dtype=float)
        self.Z = np.random.choice([i for i in range(k)], size=self.N)
        self.n_k = np.array([0.0 for i in range(k)])
    
            
    def calculate_n_k(self):
        self.n_k = np.array([0 for i in range(2)])
        for i in range(self.N):
            self.n_k[self.Z[i]] += 1
            
    def get_hyperparameters(self):
        
        self.calculate_n_k()
        beta_star = self.beta + self.n_k
        
        x_k_bar = []
        p_k_star = []
        m_k_star = []
        
        a_k_star = []
        b_k_star = []
        for k in range(self.K):
            x_k_bar.append(sum([self.X[i] for i in range(self.N) if self.Z[i] == k])/self.n_k[k])
            p_k_star.append(self.p + self.n_k[k])
            m_k_star.append((self.p*self.m + self.n_k[k]*x_k_bar)/p_k_star[k])
            
            
            a_k_star.append(self.a + self.n_k[k]/2) 
            b_k_star.append(self.b + sum([(self.X[i])**2 for i in range(self.N) if self.Z[i] == k])/2 + (self.p*self.m**2- p_k_star[k]*m_k_star[k]**2)/2)
            
        return beta_star, x_k_bar, p_k_star, m_k_star, a_k_star, b_k_star
            
    
    
        
    def estimate_hyperparameters(self):
 
        beta_star, x_k_bar, p_k_star, m_k_star, a_k_star, b_k_star = self.get_hyperparameters()
        
        
        
        for i in range(self.N):
            #Use the current values of the hyperparameters to calculate the conditional distribution of Z[i]
            
            k_i =self.Z[i]
            
            self.Z[i] = -1
            
            
            beta_star_neg_i, x_k_bar_neg_i, p_k_star_neg_i, m_k_star_neg_i, a_k_star_neg_i, b_k_star_neg_i = self.get_hyperparameters()
            
            probs = []
            
            for k in range(self.K):
                num = (self.p/p_k_star)**(1/2) * (np.exp(b_k_star - self.b)/(2*np.pi)**(self.n_k[k]/2)) * (math.gamma(a_k_star[k])/math.gamma(self.a)) *  (Beta([beta_star for i in range(self.K)])/Beta([self.beta for i in range(self.K)]))
                
                den = (self.p/p_k_star_neg_i[k])**(1/2) * (np.exp(b_k_star_neg_i[k] - self.b)/(2*np.pi)**(self.n_k[k]/2)) * (math.gamma(a_k_star_neg_i[k])/math.gamma(self.a)) *  (Beta([beta_star_neg_i[k] for i in range(self.K)])/Beta([self.beta for i in range(self.K)]))
                
                probs.append(num/den)
                
            probs = np.array(probs)
            
            # Sample new value of Z[i]
            
            self.Z[i] = np.random.choice([i for i in range(self.K)], p=probs/sum(probs))
            
        
            
            
        
if __name__=='__main__':
    
    # Define the parameters of the GMM
    k = 2
    mu = np.array([-1, 1], dtype=float)
    nu = np.array([10, 10], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    
    # Sample from the GMM
    gmm = GMM(k, mu, nu, pi)
    X, Z = gmm.sample(100)
    
    # Perform collapsed Gibbs sampling
    collapsed_gibbs = CollapsedGibbs(X, k, 0, 1, 1, 1)
    
    n_iters = 1000
    
    for i in range(n_iters):
        collapsed_gibbs.estimate_hyperparameters()
    
    plt.plot(X, collapsed_gibbs.Z, 'o')
    plt.xlabel('x')
    plt.title('Collapsed Gibbs Sampling: n_iters = ' + str(n_iters))
    
    plt.show()
        
        
        
        