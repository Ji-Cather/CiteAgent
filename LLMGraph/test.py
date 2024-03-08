from scipy.optimize import linear_sum_assignment
import numpy as np

# 定义租客、房屋和匹配关系
tenants = ['Tenant1', 'Tenant2', 'Tenant3']
houses = ['House1', 'House2', 'House3','House4']
matches = [('Tenant1', 'House2', 1), ('Tenant2', 'House1', 2), ('Tenant2', 'House2', 3), ('Tenant3', 'House1', 2)]

# 构建一个权重矩阵
# n = max(len(tenants), len(houses))
cost_matrix = np.zeros((len(tenants), len(houses)))

# 填充权重矩阵
for match in matches:
    tenant_index = tenants.index(match[0])
    house_index = houses.index(match[1])
    # 将分数作为负数填充，因为linear_sum_assignment函数是寻找最小费用的匹配
    cost_matrix[tenant_index, house_index] = -match[2]

# 应用Kuhn-Munkres算法找到最优匹配
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 准备最优匹配结果
optimal_matches = [(tenants[i], houses[j], -cost_matrix[i, j]) for i, j in zip(row_ind, col_ind)]
total_score = -cost_matrix[row_ind, col_ind].sum()

optimal_matches, total_score

print(total_score)

print(optimal_matches)