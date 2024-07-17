import numpy as np


def colebrook(Re, K):
    if Re < 2300:
        return 64 / Re
    else:
        # Colebrook方程的迭代求解
        X1 = K * Re * 0.123968186335417556
        X2 = np.log(Re) - 0.779397488455682028
        F = X2 - 0.2
        E = (np.log(X1 + F) - 0.2) / (1 + X1 + F)
        F = F - (1 + X1 + F + 0.5 * E) * E * (X1 + F) / (1 + X1 + F + E * (1 + E / 3))
        E = (np.log(X1 + F) + F - X2) / (1 + X1 + F)
        F = F - (1 + X1 + F + 0.5 * E) * E * (X1 + F) / (1 + X1 + F + E * (1 + E / 3))
        return 1.151292546497022842 / F * F


def pipe_flows_cal(N, Np, Nd, D, ep, length, mq, A, B):
    # 初始化dm
    m_pipe = np.ones(Np)
    err = 1
    pre = 0
    # 简化部分参数为常数，以减少函数需要的变量个数
    rho = 958.4  # 水在100度时的密度 kg/m^3
    g = 9.81  # 重力加速度
    viscosity = 0.294e-6  # 温度为100度时的动力粘度 m2/s

    while err > 1e-3:
        # 计算管道流量dm
        m_node = np.dot(A, m_pipe)  # 节点的流量注入
        dPhi = m_node[:Nd] - mq[:Nd]  # （负荷）节点流量偏差值
        HJ0 = A[:N - 1, :]

        # 计算流速、雷诺数等
        vel = m_pipe / (np.pi * D * D / 4) / rho  # 单位 m kg/s, V m/s
        Re = abs(vel) * D / viscosity

        factor = np.zeros(Np)
        for ire in range(Np):
            factor[ire] = colebrook(Re[ire], ep[ire] / D[ire])

        Kf = factor * length / D / (np.pi * D * D / 4) ** 2 / 2 / g / rho ** 2
        dpre = np.dot(B, Kf * abs(m_pipe) * m_pipe)  # 压力环方程
        HJpre = 2 * B * (Kf * abs(m_pipe))

        dH = np.concatenate((dPhi, dpre))
        HJ = np.concatenate((HJ0, HJpre))
        dx = -np.linalg.solve(HJ, dH)
        err = max(abs(dH))
        m_pipe += dx
        pre += 1

    return m_pipe

