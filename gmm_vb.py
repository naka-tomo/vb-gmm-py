import numpy as np
from scipy.special import psi
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM_VB():
    def __init__(self, X, K ):
        N = X.shape[0]
        D = X.shape[1]
        self.K = K
        self.N = N
        self.D = D
        self.X = X.reshape( (-1,D,1) )

        # 事前分布のパラメータ
        self.alpha0 = np.ones( K ) * 0.1
        self.beta0 = 0.1
        self.nu0 = 0.1
        self.m0 = np.zeros( (D,1) )
        self.W0 = np.eye( D,D )

        # 事後確率分布のパラメータ
        self.eta_nk = np.random.random( (N, K) )
        self.eta_nk /= np.sum( self.eta_nk,1,keepdims=True)
        self.nu_k = np.zeros( K )
        self.W_k = np.zeros( (K, D, D) )
        self.m_k = np.zeros( (K, D, 1) )
        self.beta_k = np.zeros( K )
        self.alpha_k = np.zeros( K )

    # q(s)の更新
    def update_q_s(self):
        for k in range(self.K):
            # 各期待値を計算
            exp_s = self.eta_nk
            exp_lam = self.nu_k[k]*self.W_k[k]
            exp_lam_mu = self.nu_k[k]*self.W_k[k]@self.m_k[k]
            exp_ln_lam = np.sum( [ psi(0.5*(self.nu_k[k]+1-d)) for d in range(self.D) ] ) + self.D*np.log(2) + np.log(np.linalg.det(self.W_k[k]))
            exp_mu_lam_mu = self.nu_k[k]*self.m_k[k].T@self.W_k[k]@self.m_k[k] + self.D/self.beta_k[k]

            self.alpha_k[k] = np.sum( exp_s[:,k] ) + self.alpha0[k]
            exp_pi = psi(self.alpha_k[k]) - psi(np.sum(self.alpha_k))

            for n in range(self.N):
                self.eta_nk[n,k] = np.exp( -0.5*self.X[n].T@exp_lam@self.X[n] + self.X[n].T@exp_lam_mu -0.5*exp_mu_lam_mu + 0.5*exp_ln_lam + exp_pi )

        self.eta_nk += 1e-5        # 計算安定化のため微小量足す
        self.eta_nk /= ( np.sum( self.eta_nk,axis=1,keepdims=True) ) # 正規化
        
    # q(mu)の更新
    def update_q_mu(self):
        for k in range(self.K):
            exp_s = self.eta_nk
            self.beta_k[k] = np.sum( exp_s[:,k] ) + self.beta0
            self.m_k[k] = (np.sum( [ exp_s[n,k]*self.X[n] for n in range(self.N) ], axis=0 ) + self.beta0*self.m0) / self.beta_k[k]

    # q(Λ)の更新
    def update_q_lam(self):
        for k in range(self.K):
            exp_s = self.eta_nk
            self.nu_k[k] = np.sum( exp_s[:,k] ) + self.nu0
            self.W_k[k] = np.sum( [ exp_s[n,k]*self.X[n]@self.X[n].T for n in range(self.N) ], axis=0 ) + self.beta0*self.m0@self.m0.T - self.beta_k[k]*self.m_k[k]@self.m_k[k].T +  self.W0

    def get_results(self):
        return np.argmax( self.eta_nk, axis=1 )

    def plot(self, X ):
        min_v = min( np.min(X[0:]), np.min(X[1:]) )
        max_v = max( np.max(X[0:]), np.max(X[1:]) )

        x_grid, y_grid = np.meshgrid( np.linspace(min_v, max_v, 100), np.linspace(min_v, max_v, 100) )
        points = np.stack( [x_grid.flatten(), y_grid.flatten()], axis=1 )

        classes = np.argmax( self.eta_nk, axis=1 )

        for k in range(self.K):
            mu = self.m_k[k].flatten()
            S = np.linalg.inv(self.nu_k[k]*self.W_k[k])
            probs = multivariate_normal.pdf(points, mean=mu, cov=S)
            plt.contour( x_grid, y_grid, probs.reshape( x_grid.shape ) )                    

        for k in range(self.K):
            plt.scatter( X[:,0][classes==k], X[:,1][classes==k] )

def main():
    X = np.loadtxt( "data1.txt" )

    vb = GMM_VB(X, 2)
    for it in range(10):
        vb.update_q_mu()
        vb.update_q_lam()
        vb.update_q_s()

    print( "クラスタリング結果：", vb.get_results() )
    vb.plot(X)
    plt.show()

if __name__ == '__main__':
    main()