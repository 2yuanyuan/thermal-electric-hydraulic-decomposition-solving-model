import numpy as np


def supply_tem_cal(Np, Nd, length, A, lamda, cp, Ak, m, Ts, Ta):
    Cs = np.zeros((Nd, Nd))  # 初始化
    bs = np.zeros(Nd)  # 初始化

    for i in range(Nd):  # 节点i
        numberOfOnesInRow2 = np.sum(A[i, :] == 1)  # 确认有几条支路汇入
        if numberOfOnesInRow2 == 1:  # 判断为独立节点
            Cs[i, i] = 1
            k = np.where(A[i, :] == 1)[0][0]  # 确定注入节点i的是哪条管道k
            j = np.where(A[:, k] == -1)[0][0]  # 确定和管道k、节点i连接的是哪个节点j
            bs[i] = Ts[j] * np.exp(-lamda * length[k] / cp / m[k])
        else:  # 混合节点
            K = np.where(A[i, :] == 1)[0]  # 确定注入节点i的所有管道k
            Cs[i, i] = np.sum(m[K])
            for j in range(Np):
                if j == i:
                    # i==j跳过
                    continue
                if j <= (Nd-1):  # 判断节点j是否为负荷节点
                    k = Ak[i, j]  # 确定节点i和j之间连接的管道
                    Cs[i, j] = -m[k] * np.exp(-lamda * length[k] / cp / m[k])
                else:
                    k = Ak[i, j]
                    bs[i] += m[k] * Ts[j] * np.exp(-lamda * length[k] / cp / m[k])
    # 检查 Cs 矩阵是否为奇异矩阵
    if np.linalg.det(Cs) == 0:
        print("Cs 矩阵是奇异矩阵，使用伪逆矩阵求解")
        sol1 = np.linalg.pinv(Cs).dot(bs)
    else:
        # 求解线性方程组
        sol1 = np.linalg.solve(Cs, bs)
    sol1 += Ta  # 加回环境温度
    return sol1


def return_tem_cal(Nd, length, mq, A, lamda, cp, Ak, m, To, Ta):
    Cr = np.zeros((Nd, Nd))  # 初始化
    br = np.zeros(Nd)  # 初始化

    for i in range(Nd):  # 负荷节点i
        numberOfOnesInRow2 = np.sum(-A[i, :])  # 确认有几条支路流出
        if numberOfOnesInRow2 != 0:  # 判断为独立节点
            Cr[i, i] = 1
            br[i] = To[i]
        else:  # 混合节点
            for j in range(Nd):
                if j == i:
                    # i==j跳过
                    continue
                else:
                    k = Ak[i, j]  # 确定节点i和j之间连接的管道
                    Cr[i, i] = m[i]
                    Cr[i, j] = -m[k] * np.exp(-lamda * length[k] / cp / m[k])
                    br[i] = mq[i] * To[i]

    # 检查 Cr 矩阵是否为奇异矩阵
    if np.linalg.det(Cr) == 0:
        print("Cr 矩阵是奇异矩阵，使用伪逆矩阵求解")
        sol2 = np.linalg.pinv(Cr).dot(br)
    else:
        # 求解线性方程组
        sol2 = np.linalg.solve(Cr, br)
    sol2 = sol2 + Ta  # 加回环境温度
    # 热源回水温度
    To[2] = (m[0] * (sol2[0] - Ta) * np.exp(-lamda * length[0] / cp / m[0]) +
             m[2] * (sol2[1] - Ta) * np.exp(-lamda * length[2] / cp / m[2])) / (-mq[2])
    To = To + Ta

    return sol2, To[2]
