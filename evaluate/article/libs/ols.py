import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def OLS(citation, relevance):
    """
    计算两个m*m矩阵的Beta系数
    :param A: 第一个m*m矩阵
    :param B: 第二个m*m矩阵
    :return: Beta系数矩阵
    """
    A = citation
    B = relevance
    m, n = A.shape
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
    t_statistic, p_value = stats.ttest_1samp(beta_coefficients.reshape(-1,1), 0)
    if p_value < 0.05:
        return (beta_coefficients.avg(),beta_coefficients.std())
    else:
        return None