import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 参数
q1, q2 = 1.2, 7
T1, T2 = 8, 19
k_values = [6.0, 7.4, 9.0]

def Da1(T, k):
    exp1 = np.exp(-T1 / T)
    exp2 = np.exp(-T2 / T)
    Da1 = (T - 1) / ((q1 - T + 1) * exp1 + k * (q2 - T + 1) * exp2)
    return Da1

T_vals = np.arange(0.1, 10)  # 避免 T 过小导致除零
plt.figure(figsize=(8, 6))

# # 对每个 k 值，计算对应的 Da1 数组并绘制
# for k in k_values:
#     Da1_vals = [Da1(T, k) for T in T_vals]  # 计算每个 T 对应的 Da1
#     plt.plot(Da1_vals, T_vals, label=f'k = {k}')

k = 6
Da1_vals = [Da1(T, k) for T in T_vals]
plt.plot(T_vals, Da1_vals)
plt.show()
# # 图形设置
# plt.xlabel('Da₁')
# plt.ylabel('T')
# plt.title('Steady-State Curve for Different k Values')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
