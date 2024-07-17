import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import ModelCheckpoint

# Set the path to your data directory
path = r'D:\学习\热力控制\人工智能课程'
os.chdir(path)

# Load and preprocess data
path = 'data/preprocessed/'
file_list = os.listdir(path)

# Water data
file_water = pd.read_csv(path + file_list[2])
file_water['time'] = file_water['time'].str.split('T')
file_water['MO'] = file_water['time'].apply(lambda x: int(x[0].split('-')[1]))
file_water['DY'] = file_water['time'].apply(lambda x: int(x[0].split('-')[2]))
file_water['HR'] = file_water['time'].apply(lambda x: int(x[1].split(':')[0]))
file_water.drop(['time'], axis=1, inplace=True)

# Power and cooling data
file_power = pd.read_csv(path + file_list[1])
file_power['time'] = file_power['time'].str.split('T')
file_power['MO'] = file_power['time'].apply(lambda x: int(x[0].split('-')[1]))
file_power['DY'] = file_power['time'].apply(lambda x: int(x[0].split('-')[2]))
file_power['HR'] = file_power['time'].apply(lambda x: int(x[1].split(':')[0]))
file_power.drop(['time'], axis=1, inplace=True)

# Gas (heating) data
file_gas = pd.read_csv(path + file_list[0])
file_gas['time'] = file_gas['time'].str.split('T')
file_gas['MO'] = file_gas['time'].apply(lambda x: int(x[0].split('-')[1]))
file_gas['DY'] = file_gas['time'].apply(lambda x: int(x[0].split('-')[2]))
file_gas['HR'] = file_gas['time'].apply(lambda x: int(x[1].split(':')[0]))
file_gas.drop(['time'], axis=1, inplace=True)

# Weather data
file_weather = pd.read_csv(path + 'weather.csv')
file_weather = file_weather.drop([0, 1463, 1464])
file_weather = file_weather.rename(columns={"T2MDEW": "T", "WS10M": "WS", "WD10M": "WD", "RH2M": "RH", "PS": "P"})

# Merge data
file_water_merge = pd.merge(file_water, file_weather, how='right', on=['MO', 'DY', 'HR'])
file_water_merge[['dlh', 'xlh', 'sy']] = file_water_merge[['dlh', 'xlh', 'sy']].interpolate(method='linear')
file_water_merge = file_water_merge.rename(columns={"dlh": "dlh_water", "xlh": "xlh_water", "sy": "sy_water"})

file_power_merge = pd.merge(file_power, file_weather, how='right', on=['MO', 'DY', 'HR'])
file_power_merge[['dlh_all', 'xlh_all', 'sy_all', 'xlh_under', 'dlh_under']] = file_power_merge[['dlh_all', 'xlh_all', 'sy_all', 'xlh_under', 'dlh_under']].interpolate(method='linear')

file_gas_merge = pd.merge(file_gas, file_weather, how='right', on=['MO', 'DY', 'HR'])
file_gas_merge[['dlh', 'xlh']] = file_gas_merge[['dlh', 'xlh']].interpolate(method='linear')
file_gas_merge = file_gas_merge.rename(columns={"dlh": "dlh_gas", "xlh": "xlh_gas"})

file_merge = pd.merge(pd.merge(file_power_merge, file_water_merge, how='left', on=['MO', 'DY', 'HR', 'T', 'WS', 'WD', 'RH', 'P']), file_gas_merge, how='left', on=['MO', 'DY', 'HR', 'T', 'WS', 'WD', 'RH', 'P'])

# Handle negative values
file_merge['dlh_all'][file_merge['dlh_all'] < 0] = 0

# Feature and target datasets
X = file_merge.drop(columns=['xlh_gas', 'xlh_all', 'xlh_under', 'xlh_water'])
y = file_merge[['dlh_all', 'dlh_under']]

# Normalize X
x = (X - X.min()) / (X.max() - X.min())
x[['MO', 'DY', 'HR']] = X[['MO', 'DY', 'HR']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False, random_state=666)

# CNN Model
model_CNN = Sequential([
    layers.Conv1D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2)
])

# Hyperparameters
lr = 0.0005  # Learning rate
epoch = 1500  # Epochs
model = model_CNN  # Model selection

# Create batch dataset for TensorFlow
def create_batch_dataset(X, y, train=True, buffer_size=2000, batch_size=64):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)

train_dataset, train_labels = np.array(X_train), np.array(y_train)
test_dataset, test_labels = np.array(X_test), np.array(y_test)
train_dataset = train_dataset.reshape((1242, 14, 1))
test_dataset = test_dataset.reshape((220, 14, 1))

train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# Model checkpoint
checkpoint_file = 'best_model-dlh.weights.h5'
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file, monitor='loss', mode='min', save_best_only=True, save_weights_only=True)

# Train the model
history = model.fit(train_batch_dataset, epochs=epoch, validation_data=test_batch_dataset, callbacks=[checkpoint_callback])

# Predictions
test_preds = model.predict(test_dataset, verbose=1)

# R2 score
score = r2_score(test_labels, test_preds)
print("r^2 value: ", score)

# Plot results
plt.figure(figsize=(16, 8))
plt.plot(test_labels[:, 0], label="True DLH")  # True DLH
plt.plot(test_preds[:, 0], label="Pred DLH")  # Predicted DLH
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(test_labels[:, 1], label="True DLH_Under")  # True DLH_Under
plt.plot(test_preds[:, 1], label="Pred DLH_Under")  # Predicted DLH_Under
plt.legend(loc='best')
plt.show()

# Combine extracted data with predictions
df_time = pd.DataFrame(file_merge.iloc[1242:1462, 7:9])
df_loads = pd.DataFrame(test_preds, columns=['电力负荷', '热力负荷'])
df_loads.index = range(1242, 1462)
df_combined = pd.concat([df_time, df_loads], axis=1)

# Process data for cost calculation
df_results = pd.DataFrame(columns=['时间', 'P1_source', 'P_load', '电费'])
for index, row in df_combined.iterrows():
    时间 = row['HR']
    环境温度 = row['T']
    电力负荷 = row['电力负荷']
    冷负荷 = row['热力负荷']
    热力负荷 = 冷负荷 * 0.001 * 0.5  # Convert to kW, variance factor 0.5

    # Input basic network data
    N = 3
    Np = 3
    Nd = 2
    D = np.ones(3) * 150e-3
    ep = np.ones(3) * 1.25e-3
    length = np.array([400, 400, 400])
    mq = np.zeros(Np)
    A = np.array([[1, -1, 0], [0, 1, 1], [-1, 0, -1]])
    B = np.array([[1, 1, -1]])
    lamda = 0.2
    cp = 4182
    Ak = np.array([[-1, 1, 0], [1, -1, 2], [0, 2, -1]])
   
