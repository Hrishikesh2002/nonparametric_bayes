import numpy as np
import matplotlib.pyplot as plt


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
        
    
    
            
            
class Gibbs_sampler:
    
    def __init__(self, X, k, m, p, a, b) -> None:
        self.X = X
        self.N = len(X)
        self.m = m
        self.p = p
        self.a = a
        self.b = b
        self.K = k
        self.pi = np.array([1/k for i in range(k)])
        self.beta = np.array([1 for i in range(k)])
        self.mu = np.array([0 for i in range(k)])
        self.nu = np.array([10 for i in range(k)])
        self.Z = np.random.choice([i for i in range(k)], size=self.N)
        self.n_k = np.array([0 for i in range(k)])
    
    def calculate_n_k(self):
        self.n_k = np.array([0 for i in range(self.K)])
        for i in range(self.N):
            self.n_k[self.Z[i]] += 1
        
    def estimate_Z(self):

        probs = [0 for i in range(self.K)]
        
        for i in range(self.N):
            for k in range(self.K):
                probs[k] = self.pi[k]*((self.nu[k])**(1/2))*np.exp(-self.nu[k]*((self.X[i] - self.mu[k])**2)/2)
                # print("probs: ", probs[k])
                
            probs = probs/sum(probs)
            # print(probs)
            
            self.Z[i] = np.random.choice([i for i in range(self.K)], p=probs)
            
    def estimate_pi(self):
        beta_star = self.beta + self.n_k
        self.pi = np.random.dirichlet(beta_star)
    

    def estimate_mu_nu(self):
        for k in range(self.K):
            x_k_bar = sum([self.X[i] for i in range(self.N) if self.Z[i] == k])/self.n_k[k]
            p_k_star = self.p + self.n_k[k]
            m_k_star = (self.p*self.m + self.n_k[k]*x_k_bar)/p_k_star
            # print("p_k_star: ", p_k_star)
            # print("m_k_star: ", m_k_star)
            # print("x_k_bar: ", x_k_bar)
            
            self.mu[k] = np.random.normal(m_k_star, 1/np.sqrt(p_k_star*self.nu[k]))
            
            a_k_star = self.a + self.n_k[k]/2
            b_k_star = self.b + sum([(self.X[i])**2 for i in range(self.N) if self.Z[i] == k])/2 + (self.p*self.m**2)/2 - p_k_star*m_k_star**2
            
            # print("a_k_star: ", a_k_star)
            # print("b_k_star: ", b_k_star)
            # self.nu[k] = np.random.gamma(a_k_star, 1/b_k_star)
            
    def reorder_mu(self):
        for i in range(self.K):
            for j in range(i+1, self.K):
                if self.mu[i] > self.mu[j]:
                    self.mu[i], self.mu[j] = self.mu[j], self.mu[i]
                    self.nu[i], self.nu[j] = self.nu[j], self.nu[i]
                    self.pi[i], self.pi[j] = self.pi[j], self.pi[i]
                    self.n_k[i], self.n_k[j] = self.n_k[j], self.n_k[i]
                    self.beta[i], self.beta[j] = self.beta[j], self.beta[i]
                    for n in range(self.N):
                        if self.Z[n] == i:
                            self.Z[n] = j
                        elif self.Z[n] == j:
                            self.Z[n] = i
            
    
    def estimate_parameters(self, n_iters):
        for i in range(n_iters):
            self.calculate_n_k()
            self.estimate_pi()
            self.estimate_mu_nu()
            # self.print_parameters()
            self.estimate_Z()
            
        self.reorder_mu()
         
            
    
    def print_parameters(self):
        print("pi: ", self.pi)
        print("mu: ", self.mu)
        print("nu: ", self.nu)
    
    def print_Z(self):
        print("Z: ", self.Z)
        
            
            
            
def calculate_accuracy(Z1, Z2):
    acc = 0
    for i in range(len(Z1)):
        if Z1[i] == Z2[i]:
            acc += 1
    return acc/len(Z1)
        
        
        

if __name__ == "__main__":
    
    #generate data
    
    gmm = GMM(k=3, mu=[0, 5, 20], nu=[1, 1, 1], pi=[1/3, 1/3, 1/3])
    X, Z = gmm.sample(50)
    
    print("X: ", X)
    gmm.print_parameters()
    print("Cluster assignments according to the true model: ", Z)
        
    #Use Gibbs sampling to estimate the parameters of the GMM
    sampler = Gibbs_sampler(X, k=3, m=0, p=1, a=100, b=100)
    sampler.nu = np.array([1, 1, 1])
    
    # sampler.estimate_parameters(7000)
    # sampler.print_parameters()
    # sampler.print_Z()

    
    
    #Plot the data after every 100 iterations
    
    for i in range(0, 1000, 100):
        sampler.estimate_parameters(100)
        plt.scatter(X, [0 for i in range(len(X))], c=sampler.Z)
        plt.title("Iteration: " + str(i))
        plt.xlabel("X")
        plt.show()
        
    print("Cluster assignments according to the estimated model: ", sampler.Z)
    
    print("Accuracy: ", calculate_accuracy(Z, sampler.Z))
    
    
    
    
