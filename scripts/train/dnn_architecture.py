import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, regularizers
from keras.metrics import MeanSquaredError
from keras.utils import register_keras_serializable
import tqdm
import os
import json

# Create and save models
def architecture(x):
    if x not in [1, 2, 3, 4, 5, 6, 7]:
        raise ValueError("Input parameter must be 1, 2, or 3.")
    # 24 K-bytes
    if x == 1:
        model= models.Sequential([
            layers.Reshape((1080, 1)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=8), 

            layers.Flatten(),
            layers.Dense(1, kernel_regularizer=regularizers.l2(0.0001)), 
            #layers.Dense(1)
            ])
        return model
    # 66 K_bytes
    elif x == 2:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(8, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(), 
            layers.Dense(1)
        ])
        return model
    # 192 K_bytes
    elif x == 3:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(8, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 326 K_bytes
    elif x == 4:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(16, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 596 K_bytes
    elif x == 5:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(8, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(16, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 1112.54 k_bytes
    elif x == 6:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(8, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(32, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 2193.90 k_bytes
    elif x == 7:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(16, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(32, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    
    else:
        raise ValueError("Input parameter must be 1, 2, or 3.")

store_mse = {f'model{i}_{j}': 0 for i in range(1, 4) for j in range(1, 8)}

for i in range(1,6):
    for trial in range(3):
        model1 = architecture(i)
        model1.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
        model1.build((32,1080))
        model1.summary()
        model1.fit(np.array(xTrain), np.array(yTrain1), epochs=13, verbose=1)
        loss, mse = model1.evaluate(xTest, yTest1, verbose=1)
        store_mse[f'model1_{i}'] += mse
        print(f'model1_{i} MSE: {mse}')
        model1.save(f"./models/lidar_model1_{i}.keras")

        model2 = architecture(i)
        model2.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
        model2.build((32,1080))
        model2.summary()
        model2.fit(np.array(xTrain), np.array(yTrain2), epochs=13, verbose=1)
        loss, mse = model2.evaluate(xTest, yTest2, verbose=1)
        store_mse[f'model2_{i}'] += mse
        print(f'model2_{i} MSE: {mse}')
        model2.save(f"./models/lidar_model2_{i}.keras")

        model3 = architecture(i)
        model3.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
        model3.build((32,1080))
        model3.summary()
        model3.fit(np.array(xTrain), np.array(yTrain3), epochs=13, verbose=1)
        loss, mse = model3.evaluate(xTest, yTest3, verbose=1)
        store_mse[f'model3_{i}'] += mse
        print(f'model3_{i} MSE: {mse}')
        model3.save(f"./models/lidar_model3_{i}.keras")

average_mse = {}
for x in store_mse:
    average_mse[x] = store_mse[x] / 2  # Assuming each model is evaluated 5 times
    print(f'Average MSE for {x}: {average_mse[x]}')

for i in range(1,4):
    for j in range(1,8):
        file_path = f'models/lidar_model{i}_{j}.keras'
        file_size = os.path.getsize(file_path)
        print(f"Model {i}_{j} file size :", file_size/1024, "k_bytes")

with open('mse_data.json', 'w') as f:
    json.dump(average_mse, f)

print(average_mse)

