__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 

from scipy import stats


from scipy.stats import ttest_ind
from evaluate.article.libs import utils


#######################################################################
# QAP
#######################################################################
class QAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, X=None, npermutations=-1, diagonal=False,type = 'pearson'):
        '''
        Initialization of variables
        :param Y: numpy array depended variable
        :param X: numpy array independed variable
        :return:
        '''
        self.Y = Y
        self.X = X
        self.npermutations = npermutations
        self.diagonal = diagonal
        self.beta = None
        self.Ymod = None
        self.betas = []
        self.type = type

    def init(self):
        '''
        Shows the correlation of the initial/original variables (no shuffeling)
        :return:
        '''
        self.beta = self.correlation(self.X, self.Y)
        self.stats(self.X, self.Y)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def qap(self):
        '''
        Quadratic Assignment Procedure
        :param npermutations:
        :return:
        '''
        self.init()
        self._shuffle()

    def _shuffle(self):
        self.Ymod = self.Y.copy()
        for t in range(self.npermutations):
            self._rmperm()
            data = self.correlation(self.X, self.Ymod, False)
            if data is not None:
                self._addBeta(data)

    def correlation(self, x, y, show=True):
        '''
        Computes Pearson's correlation value of variables x and y.
        Diagonal values are removed.
        :param x: numpy array independent variable
        :param y: numpu array dependent variable
        :param show: if True then shows pearson's correlation and p-value.
        :return:
        '''
        if self.type == 'pearson':
            if not self.diagonal:
                xflatten = np.delete(x, [i*(x.shape[0]+1)for i in range(x.shape[0])])
                yflatten = np.delete(y, [i*(y.shape[0]+1)for i in range(y.shape[0])])
                pc = pearsonr(xflatten, yflatten)
            else:
                pc = pearsonr(x.flatten(), y.flatten())
            if show:
                utils.printf('Pearson Correlation: {}'.format(pc[0]))
                utils.printf('p-value: {}'.format(pc[1]))
            if pc[1] > 0.05 and not show:
                return None
            return pc
        elif self.type == 'ols':
            #ols
            results = self.ols(x,y)
            if results.pvalues[1]>0.05:
                return None
            return results.params[1],results.pvalues[1]

            # betas,p =self.count_ols(x,y)
            # if p> 0.05 and not show:
            #     return None
            # return betas
    #####################################################################################
    # Handlers
    #####################################################################################

    def _addBeta(self, p):
        '''
        frequency dictionary of pearson's correlation values
        :param p: person's correlation value
        :return:
        '''
        try:
            p = round(p[0],6)
        except:
            p = round(p,6)
        self.betas.append(p)

    def _rmperm(self):
        shuffle = np.random.permutation(self.Ymod.shape[0])
        np.take(self.Ymod,shuffle,axis=0,out=self.Ymod)
        np.take(self.Ymod,shuffle,axis=1,out=self.Ymod)


    #####################################################################################
    # Plots & Prints
    #####################################################################################

    def summary(self):
        utils.printf('')
        utils.printf('# Permutations: {}'.format(self.npermutations))
        utils.printf('Correlation coefficients: Obs. Value({}), Significance({})'.format(self.beta[0], self.beta[1]))
        utils.printf('')
        utils.printf('- Sum all betas: {}'.format(sum(self.betas)))
        utils.printf('- Min betas: {}'.format(min(self.betas)))
        utils.printf('- Max betas: {}'.format(max(self.betas)))
        utils.printf('- Average betas: {}'.format(np.average(self.betas)))
        utils.printf('- Std. Dev. betas: {}'.format(np.std(self.betas)))
        utils.printf('')
        utils.printf('prop >= {}: {}'.format(self.beta[0], sum([1 for b in self.betas if b >= self.beta[0] ])/float(len(self.betas))))
        utils.printf('prop <= {}: {} (proportion of randomly generated correlations that were as {} as the observed)'.format(self.beta[0], sum([1 for b in self.betas if b <= self.beta[0] ])/float(len(self.betas)), 'large' if self.beta[0] >= 0 else 'small'))
        utils.printf('')

    def plot(self):
        '''
        Plots frequency of pearson's correlation values
        :return:
        '''
        plt.hist(self.betas)
        plt.xlabel('regression coefficients')
        plt.ylabel('frequency')
        plt.title('QAP')
        plt.grid(True)
        plt.savefig('qap.pdf')
        plt.show()
        plt.close()

    #####################################################################################
    # Others
    #####################################################################################

    def stats(self, x, y):
        if not self.diagonal:
            xflatten = np.delete(x, [i*(x.shape[0]+1)for i in range(x.shape[0])])
            yflatten = np.delete(y, [i*(y.shape[0]+1)for i in range(y.shape[0])])
            p = np.corrcoef(xflatten,yflatten)
            utils.printf('Pearson\'s correlation:\n{}'.format(p))
            utils.printf('Z-Test:{}'.format(ztest(xflatten, yflatten)))
            utils.printf('T-Test:{}'.format(ttest_ind(xflatten, yflatten)))
        else:
            p = np.corrcoef(x, y)
            utils.printf('Pearson\'s correlation:\n{}'.format(p))
            utils.printf('Z-Test:{}'.format(ztest(x, y)))
            utils.printf('T-Test:{}'.format(ttest_ind(x, y)))

    def ols(self, x, y):
        xflatten = np.delete(x, [i*(x.shape[0]+1)for i in range(x.shape[0])])
        yflatten = np.delete(y, [i*(y.shape[0]+1)for i in range(y.shape[0])])
        xflatten = sm.add_constant(xflatten)
        model = sm.OLS(yflatten,xflatten)
        results = model.fit()
        return results
    
    def count_ols(self, x, y):
        xflatten = np.array(x)
        yflatten = np.array(y)
        
        m, n = xflatten.shape
        for i in range(m):
            xflatten[i][i] = 0
            yflatten[i][i] = 0
        
        A = yflatten
        B = xflatten
        beta_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                # 将当前行和列展平成一维数组
                a = A[i, :].reshape(-1, 1)
                b = B[:, j].reshape(-1, 1)
                
                # 使用statsmodels进行线性回归
                X = sm.add_constant(a)  # 增加常数项
                model = sm.OLS(b, X).fit()
                
                # 得到Beta系数
                beta_matrix[i, j] = model.params[1]  # 取第一个变量的系数
        beta_coefficients = beta_matrix
        t_statistic, p_value = stats.ttest_1samp(beta_matrix.reshape(-1,1), 0)

        return np.average(beta_coefficients),p_value


if __name__ == "__main__":
    import networkx as nx
    import numpy as np
    len_c = 20
    degree = 3
    # A = np.ones((len_c, len_c))
    
    p = degree/len_c
    G = nx.erdos_renyi_graph(len_c, p)
    A = nx.adjacency_matrix(G).toarray()
    
    B = np.ones((len_c, len_c))
    
    model = QAP(B,A,npermutations=1000,type='ols')
    model.qap()
    beta,std = np.average(model.betas), np.array(model.betas).std()
    print(beta,std)