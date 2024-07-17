import numpy as np
from hydraulic_module import pipe_flows_cal
from thermal_module import supply_tem_cal
from thermal_module import return_tem_cal
from electric_module import create_power_system
import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf#这里有小报错但是不影响运行
from tensorflow.keras import Sequential, layers, utils, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import warnings
warnings.filterwarnings('ignore')
#我的附件里人工智能课程文件，解压到一个位置，记住地址，如果在colab上就改成上面的地址，具体可询问GAI助手
path = r'D:\学习\热力控制\人工智能课程'#这里要改为你电脑上的地址
os.chdir(path)

#文件路径，文件列表
path='data/preprocessed/'
file_list=os.listdir(path)
print(file_list)

#水数据
file=pd.read_csv(path+file_list[2])
file['time']=file['time'].str.split('T')
file['MO']=pd.Series([int(x[0].split('-')[1]) for x in file['time']])
file['DY']=pd.Series([int(x[0].split('-')[2]) for x in file['time']])
file['HR']=pd.Series([int(x[1].split(':')[0]) for x in file['time']])
file.drop(['time'],axis=1,inplace=True)
file_water=file
file_water

#电力和制冷（cop转换）数据
file=pd.read_csv(path+file_list[1])
file['time']=file['time'].str.split('T')
file['MO']=pd.Series([int(x[0].split('-')[1]) for x in file['time']])
file['DY']=pd.Series([int(x[0].split('-')[2]) for x in file['time']])
file['HR']=pd.Series([int(x[1].split(':')[0]) for x in file['time']])
file.drop(['time'],axis=1,inplace=True)
file_power=file
file_power

# 气（供热）数据
file=pd.read_csv(path+file_list[0])
file['time']=file['time'].str.split('T')
file['MO']=pd.Series([int(x[0].split('-')[1]) for x in file['time']])
file['DY']=pd.Series([int(x[0].split('-')[2]) for x in file['time']])
file['HR']=pd.Series([int(x[1].split(':')[0]) for x in file['time']])
file.drop(['time'],axis=1,inplace=True)
file_gas=file
file_gas

# 天气数据
file_weather=pd.read_csv(path+'weather.csv')
file_weather=file_weather.drop(0)
file_weather=file_weather.drop(1463)
file_weather=file_weather.drop(1464)
file_weather=file_weather.rename(columns={"T2MDEW": "T", "WS10M": "WS", "WD10M": "WD", "RH2M": "RH", "PS": "P"})
file_weather

file_water.duplicated().value_counts() #无重复值
file_power.duplicated().value_counts() #无重复值
file_gas.duplicated().value_counts() #无重复值
file_water.describe()
file_power.describe()
file_gas.describe()

file_water_merge=pd.merge(file_water,file_weather,how='right',on=['MO','DY','HR'])
file_water_merge[['dlh','xlh','sy']]=file_water_merge[['dlh','xlh','sy']].interpolate(method='linear')
file_water_merge=file_water_merge.rename(columns={"dlh": "dlh_water", "xlh": "xlh_water", "sy": "sy_water"})

file_water_merge

file_power_merge=pd.merge(file_power,file_weather,how='right',on=['MO','DY','HR'])
file_power_merge[['dlh_all','xlh_all','sy_all','xlh_under','dlh_under']]=file_power_merge[['dlh_all','xlh_all','sy_all','xlh_under','dlh_under']].interpolate(method='linear')

file_power_merge

file_gas_merge=pd.merge(file_gas,file_weather,how='right',on=['MO','DY','HR'])
file_gas_merge[['dlh','xlh']]=file_gas_merge[['dlh','xlh']].interpolate(method='linear')
file_gas_merge=file_gas_merge.rename(columns={"dlh": "dlh_gas", "xlh": "xlh_gas"})

file_gas_merge

file_pass=pd.merge(file_power_merge,file_water_merge,how='left',on=['MO','DY','HR','T','WS','WD','RH','P'])
file_merge=pd.merge(file_pass,file_gas_merge,how='left',on=['MO','DY','HR','T','WS','WD','RH','P'])

#负值判断
file_merge.mask(file_merge<0).isnull().any()

#将负值替换为0
file_merge['dlh_all'][file_merge['dlh_all']<0]=0

file_merge['dlh_all'].plot()

#负值判断
file_merge.mask(file_merge<0).isnull().any()

file_merge
# 提取第七列（索引为6）和第1242行到第1461行（索引从1241到1460），获得时间
extracted_data_iloc = file_merge.iloc[1242:1462, 7:9]  # 注意，索引是从0开始

#print(extracted_data_iloc)

file_merge['dlh_all'].plot() #蓝
file_merge['xlh_all'].plot() #黄
file_merge['sy_all'].plot() #绿

file_merge['dlh_under'].plot() #蓝
file_merge['xlh_under'].plot() #黄

file_merge['dlh_water'].plot() #蓝
file_merge['xlh_water'].plot() #黄
file_merge['sy_water'].plot() #绿

file_merge['dlh_gas'].plot() #蓝
file_merge['xlh_gas'].plot() #黄

# 特征数据集
X = file_merge
X = X.drop(columns=['xlh_gas'], axis=1)
X = X.drop(columns=['xlh_all'], axis=1)
X = X.drop(columns=['xlh_under'], axis=1)
X = X.drop(columns=['xlh_water'], axis=1)  #删去小莲花相关特征数据

y = file_merge[['dlh_all','dlh_under']]    #大莲花的制冷和电力负荷

X.max()

X.min()

X

#归一化 X
x=(X-X.min())/(X.max()-X.min())
x['MO']=X['MO']
x['DY']=X['DY']
x['HR']=X['HR']
x

y

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False, random_state=666)
#训练集和测试集划分

#CNN模型：卷积层+汇聚层+三个全连接层
model_CNN = Sequential([
    layers.Conv1D(64,3, activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2)
])

#超参数
lr=0.0005 #学习率
epoch = 1500 #迭代数
model = model_CNN #模型选择(model_MLP, model_CNN, )

checkpoint_file = "best_model-dlh.hdf5" #权重保存命名

#训练集和测试集的数据格式转换-用于tensorflow引擎训练
def create_batch_dataset(X, y, train=True, buffer_size=2000, batch_size=64):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) # 数据封装，tensor类型
    if train: # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # 测试集
        return batch_data.batch(batch_size)
    
if model == model_CNN:
  # 构造训练特征数据集
  train_dataset, train_labels = np.array(X_train), np.array(y_train)
  # 构造测试特征数据集
  test_dataset, test_labels = np.array(X_test), np.array(y_test)
  train_dataset=train_dataset.reshape((1242, 14,1))
  test_dataset=test_dataset.reshape((220, 14,1))#220属于0.85测试集情况下的特俗情况，非通用。
  # 训练批数据
  train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
  # 测试批数据
  test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

#时序数据集划分
def create_dataset(X, y, seq_len=10):
    features = []
    targets = []

    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i+seq_len] # 序列数据
        label = y.iloc[i+seq_len] # 标签数据
        # 保存到features和labels
        features.append(data)
        targets.append(label)

    # 返回
    return np.array(features), np.array(targets)

checkpoint_file = 'best_model-dlh.weights.h5'
op = tf.keras.optimizers.Adam(learning_rate=lr) #选择adam优化器
model.compile(optimizer='adam',loss='mse')  #模型编译
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,

                                      monitor='loss',
                                      mode='min',
                                      save_best_only=True,
                                      save_weights_only=True) #模型保存
# 模型训练
history = model.fit(train_batch_dataset,
                    epochs=epoch,
                    validation_data=test_batch_dataset,
                    callbacks=[checkpoint_callback])
# 显示训练结果
#plt.figure(figsize=(16,8))
#plt.plot(history.history['loss'], label='train loss')
#plt.plot(history.history['val_loss'], label='val loss')
#plt.legend(loc='best')
#plt.show()

test_preds = model.predict(test_dataset, verbose=1)

# 计算r2值
score = r2_score(test_labels, test_preds)
print("r^2 值为： ", score)

# 绘制预测与真值结果
#plt.figure(figsize=(16,8))
#plt.plot(test_labels[:,0], label="True DLH") #真实大莲花电力负荷
#plt.plot(test_preds[:,0], label="Pred DLH") #预测大莲花电力负荷
#plt.plot(test_labels[:,1], label="True DLH_Under") #真实大莲花制冷负荷
#plt.plot(test_preds[:,1], label="Pred DLH_Under") #预测大莲花制冷负荷
#plt.legend(loc='best')
#plt.show()
#print(test_preds)
#生成一个时间、温度、电力负荷、热力负荷的表格，方便后续计算
df_time=pd.DataFrame(extracted_data_iloc)
df_loads=pd.DataFrame(test_preds,columns=['电力负荷', '热力负荷'])
df_loads.index = range(1242, 1462)
assert len(df_time) == len(df_loads)
df_combined = pd.concat([df_time, df_loads], axis=1)
#print(df_combined)

# 绘制 预测与真值结果

#plt.figure(figsize=(16,8))
#plt.plot(test_labels[:,0], label="True DLH") #真实大莲花电力负荷
#plt.plot(test_preds[:,0], label="Pred DLH")  #预测大莲花电力负荷
#plt.legend(loc='best')
#plt.show()

# 绘制 预测与真值结果
#plt.figure(figsize=(16,8))
#plt.plot(test_labels[:,1], label="True DLH_Under")  #真实大莲花制冷负荷
#plt.plot(test_preds[:,1], label="Pred DLH_Under")   #预测大莲花制冷负荷
#plt.legend(loc='best')
#plt.show()
#对表格之中的每一组数据进行遍历，从而得到每一个数据时刻都电价
df_results = pd.DataFrame(columns=['时间', 'P1_source', 'P_load', '电费'])
for index, row in df_combined.iterrows():
    序号 = row[0]
    时间 = row['HR']  # 确保这里使用的数据框中的实际列名
    环境温度 = row['T']  # 确保这里使用的数据框中的实际列名
    电力负荷 = row['电力负荷']
    冷负荷 = row['热力负荷']  # 确保这里使用的数据框中的实际列名
    热力负荷 = 冷负荷*0.001*0.5#单位转换为kw，变异系数为0.5
    # 输入管网基本数据
    N = 3  # 热网节点总数
    Np = 3  # 管道的数量
    Nd = 2  # 负荷节点的数量
    D = np.ones(3) * 150e-3  # 管道直径
    ep = np.ones(3) * 1.25e-3  # 管道粗糙度
    length = np.array([400, 400, 400])  # 管道长度
    mq = np.zeros(Np)  # 初始化mq矩阵

    A = np.array([[1, -1, 0], [0, 1, 1], [-1, 0, -1]])  # 网络节点-弧关联矩阵
    B = np.array([[1, 1, -1]])  # 环路关联矩阵，必须是二维的

    lamda = 0.2  # 管道导热系数 W/(m*K)
    cp = 4182  # 水的比热容 J/(kg*K)
    Ak = np.array([[-1, 1, 0], [1, -1, 2], [0, 2, -1]])  # 检索两节点间连接的管道编号
    Ts = np.zeros(Np)  # 节点供水温度初始化
    Ts[:] = 100  # 给定热源供水温度 100 ℃，假设负荷初始的供水温度100 ℃
    To = np.zeros(Np)  # 节点回水温度计算初始化
    To[:2] = 50  # 给定节点回水温度 50 ℃
    Ta = 环境温度  # 由数据可得的实际环境温度
    Tr = np.ones(Np)

    """
    输入热负荷数据(由负荷预测得到)，开始水力-热力-电力模型的分解计算
    水力-热力-电力模型的分解求解：可参考课本流程图2.8和图3.10（并网模式）
    此处仅为一个时刻的负荷，请改为24小时（24个时刻）的负荷数据，计算每个时刻的水、热、电结果
    负荷预测结果可选择从奥体中心案例的结果手动选取输入、负荷预测代码直接接入等方法
    此处热负荷用奥体中心案例中的冷负荷代替，此处两个负荷大小视为相等。（如果负荷不相等，请做对应修改）
    """
    Phi = np.zeros(Np)
    Phi[0] = Phi[1] =  热力负荷 # 负荷节点的热负荷输入，单位为MW，两个热负荷视为相等

    # 第一次迭代时，假设的mq矩阵（初始化）
    for i in range(Nd):
        mq[i] = Phi[i] / (cp*1e-6) / (Ts[i]-To[i])

    # 初始误差设定、初始化
    dm_pipe = 1  # dm_pipe为管段流量迭代误差
    dTs_load = 1  # dTs_load为管段流量迭代误差
    m_pipe_final = np.ones(Np)
    Ts_load_final = np.zeros(Nd) + 100

    # 首先进行水力、热力相互迭代计算
    while dm_pipe > 1e-3 or dTs_load > 1e-3:
        Ts = Ts - Ta  # 减去环境温度
        To = To - Ta  # 减去环境温度
        m_pipe = pipe_flows_cal(N, Np, Nd, D, ep, length, mq, A, B)
        dm_pipe = max(abs(m_pipe - m_pipe_final))
        mq = np.dot(A, m_pipe)
        Ts[:Nd] = supply_tem_cal(Np, Nd, length, A, lamda, cp, Ak, m_pipe, Ts, Ta)
        Tr, To[Nd:] = return_tem_cal(Nd, length, mq, A, lamda, cp, Ak, m_pipe, To, Ta)
        dTs_load = max(abs(Ts[:Nd] - Ts_load_final))
        for i in range(1, Np-Nd+1):
            Ts[-i:] += Ta
        for i in range(Nd):
            To[i] += Ta
        for i in range(Nd):
            mq[i] = Phi[i] / (cp * 1e-6) / (Ts[i] - To[i])
        mq[2] = -mq[0] - mq[1]
        m_pipe_final = m_pipe
        Ts_load_final = Ts[:Nd]

    Q1_source = cp * np.abs(mq[2]) * (Ts[2] - To[2]) * 1e-6  # 热电联产机组的热出力（即热源输出功率）
    # 电力系统计算
    c = 1.2  # 热电联产机组的热电比
    P1_source = Q1_source / c  # 热电联产机组的电出力（即电源输出功率）
    #print(Q1_source, P1_source)

    """
    电力负荷由负荷预测得到，此处两个负荷大小视为相等。（如果负荷不相等，请做对应修改）
    此处仅为一个时刻的负荷示例，请改为24小时（24个时刻）的负荷数据
    负荷预测结果可选择从奥体中心案例的结果手动选取输入、负荷预测代码直接接入等方法
    """
    P_load = 电力负荷*0.001  # 此处为预测的电力负荷
    net = create_power_system(P1_source, P_load)

    # 运行潮流计算
    pp.runpp(net, tolerance_mva=1e-3)  # 这里设置容差为0.1 MVA

    # 输出水热结果
    #print("支路1流量：{:.4f}\n支路2流量：{:.4f}\n支路3流量：{:.4f}".format(m_pipe_final[0], m_pipe_final[1], m_pipe_final[2]))
    #print('节点1供水温度：{:.4f}\n节点2供水温度：{:.4f}'.format(Ts[0], Ts[1]))
    #print('节点1回水温度：{:.4f}\n节点2回水温度：{:.4f}'.format(Tr[0], Tr[1]))
    #print('热源回水温度：{:.4f}'.format(To[2]))

    # 输出潮流计算结果
    #print('母线电压:')
    for idx, bus in net.bus.iterrows():
        voltage_magnitude = net.res_bus.vm_pu.at[idx]
        voltage_angle = net.res_bus.va_degree.at[idx]
        #print(f"节点 {bus['name']}: 电压幅值 = {voltage_magnitude:.4f} p.u., 电压相角 = {voltage_angle:.4f} 度")
   # print('节点功率:')
    for idx, bus in net.bus.iterrows():
        p_mw = net.res_bus.p_mw.at[idx]
        q_mvar = net.res_bus.q_mvar.at[idx]
        #print(f"节点 {bus['name']}: 有功功率 = {p_mw:.4f} MW, 无功功率 = {q_mvar:.4f} MVar")
    # 输出线路的功率
   # print('线路功率:')
    for idx, line in net.line.iterrows():
        p_from_mw = net.res_line.p_from_mw.at[idx]
        q_from_mvar = net.res_line.q_from_mvar.at[idx]
        p_to_mw = net.res_line.p_to_mw.at[idx]
        q_to_mvar = net.res_line.q_to_mvar.at[idx]
        #print(f"线路 {line['name']} (从 {line['from_bus']} 到 {line['to_bus']}):")
        #print(f"从端有功功率 = {p_from_mw:.4f} MW, 从端无功功率 = {q_from_mvar:.4f} MVar")
        #print(f"到端有功功率 = {p_to_mw:.4f} MW, 到端无功功率 = {q_to_mvar:.4f} MVar")

    '''
    由电力系统模型计算得到未来24个小时的够购/售电情况
    此处仅为一个时刻的计算示例，请查询工业分时电价，计算一整天的总购/售电成本或收益
    '''
    # 系统购/售电情况，即节点4的功率，功率为正时，系统卖电给上级电网
    p_net = -net.res_bus.p_mw.at[3] * 1000  # 单位由MW换算成kW,此处计算结果均为负值，说明系统需要向电网买电
    if 9<=时间<11 or 15<=时间<17:#根据查询到的杭州工业电价，分时段进行计算
        p_price = 0.1995  # 电价（元/kWh），每天不同时段的价格不同
    if 时间==8 or 13<=时间<15 or 17<=时间<22:
        p_price = 0.1653
    else:
        p_price = 0.04628
    p_profit = p_net * p_price
    df_new_row = pd.DataFrame({
        '时间': [时间],
        'P1_source': [P1_source],
        'P_load': [P_load],
        '电费': [p_profit],
        'p_net':[p_net]
    })
    #使用 pd.concat 将新行数据添加到数据框
    df_results = pd.concat([df_results, df_new_row], ignore_index=True)
print(df_results)
# 定义绘制函数，一共220个数据，前23个数据生成一个图，后每24个数据生成一张图
font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
# 检查系统中是否有SimHei字体
if fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    print("SimHei.ttf font not found on the system. Please make sure the font is installed.")
# 定义绘制p_net函数
def plot_p_net_multiple(df, title, filename):
    plt.figure(figsize=(10, 5))
    lines = []

    # 每24组数据生成一条曲线
    for i in range(0, len(df), 24):
        if i + 24 <= len(df):
            segment = df.iloc[i:i+24, :]
            line, = plt.plot(segment['时间'], segment['p_net'], marker='o', label=f'第{i//24+1}段')
            lines.append(line)

    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('p_net (kW)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_electricity(df, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(df['时间'], df['电费'], marker='o')
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('电费 (元)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# 前23个数据单独生成一个图像
df_first_23 = df_results.iloc[:23, :]
plot_electricity(df_first_23, '前23小时电费波动图', '电费波动图_前23小时.png')

# 生成每24个数据的图像
for i in range(23, len(df_results) - 5, 24):
    df_segment = df_results.iloc[i:i+24, :]
    plot_electricity(df_segment, f'电费波动图_时间段_{i}_{i+24}', f'电费波动图_时间段_{i}_{i+24}.png')
df_p_net_filtered = df_results.iloc[23:-5, :]
plot_p_net_multiple(df_p_net_filtered, 'p_net 随时间变化图（每24组数据一条曲线）', 'p_net_随时间变化图.png')