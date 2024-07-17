import pandapower as pp
import numpy as np


def create_power_system(p_source, p_load):
    """
    创建电力系统网络，包含发电机、负荷和线路。
    """
    # 创建空的 pandapower 网络
    net = pp.create_empty_network()

    # 创建母线 (buses)
    b1 = pp.create_bus(net, vn_kv=11, name="Bus 1")  # 电源1 (PV bus)
    b2 = pp.create_bus(net, vn_kv=11, name="Bus 2")  # 负荷1 (PQ bus)
    b3 = pp.create_bus(net, vn_kv=11, name="Bus 3")  # 负荷2 (PQ bus)
    b4 = pp.create_bus(net, vn_kv=11, name="Bus 4")  # 上级电网 (Slack bus)

    # 创建发电机 (gen)
    pp.create_gen(net, b1, p_mw=p_source, vm_pu=1.05, name="Generator")

    # 创建负荷 (loads)
    p_factor = 0.95  # 功率因素PF，用于计算无功功率
    q_load = p_load * np.sqrt((1 / p_factor / p_factor) - 1)  # q_load为无功功率，p_load为有功功率
    pp.create_load(net, b2, p_mw=p_load, q_mvar=q_load, name="Load 1")
    pp.create_load(net, b3, p_mw=p_load, q_mvar=q_load, name="Load 2")

    # 创建外部电网 (ext_grid)
    pp.create_ext_grid(net, b4, vm_pu=1.02, va_degree=0, name="External Grid")

    # 基准阻抗计算
    base_voltage = 11  # kV
    base_power = 1  # MVA
    base_impedance = (base_voltage ** 2) / base_power  # 计算基准阻抗 (Ω)

    # 将阻抗标幺值转换为实际值
    r_per_km_pu = 0.09  # 标幺值
    x_per_km_pu = 0.1577  # 标幺值
    r_per_km = r_per_km_pu * base_impedance / 5  # 实际值 (Ω/km)
    x_per_km = x_per_km_pu * base_impedance / 5  # 实际值 (Ω/km)

    # 创建线路 (lines)
    pp.create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=r_per_km, x_ohm_per_km=x_per_km,
                                   c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, b2, b3, length_km=1, r_ohm_per_km=r_per_km, x_ohm_per_km=x_per_km,
                                   c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, b3, b4, length_km=1, r_ohm_per_km=r_per_km, x_ohm_per_km=x_per_km,
                                   c_nf_per_km=0, max_i_ka=1)

    return net
