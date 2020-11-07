# ファイルナンバリング
number  = 'model002'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import keras
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import layers 
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv

while True:
    
    url = 'http://iotai.kd.sus.ac.jp/AIContest/download.php?from=2020/5/5%2010:40:00&to='
    # url = 'http://iotai.kd.sus.ac.jp/AIContest/download.php?'

    try:
        r = requests.get(url)
        with open('./datas/data_latest.csv', mode='w') as f:
            f.write(r.text)
            print('Data has been written to data.csv')
    except requests.exceptions.RequestException as err:
        print(err)



    data = pd.read_csv('./datas/data_latest.csv')


    edited_data = data[[
    'date', 'zeniba',
    'haregamine', 'seikaen', 'kagamiko', 'kanazawa', 'osawa', 'shiyakusho', 'izuminohoikuen', 
    'tamagawahoikuen', 'hakubutsukan', 'kohigashihoikuen', 'yonezawahoikuen', 'kashiwabarakominkan',
    'kirigamine', 'shirakabako', 'kitayatsugatake', 'okutateshina', 'minoto', 'haramura', 'toyokanko', 'tokyu',
    'alpico', 'village', 'mitsuinomori', 'kajima'
    ]]


    # date列は除く
    # np_data = data.iloc[:, 1:].values
    np_data = edited_data.iloc[:, 1:].to_numpy(dtype=float)
    mmscaler = MinMaxScaler(feature_range=(0,1)) # インスタンスの作成
    mmscaler.fit(np_data)           # 最大・最小を計算
    np_data = mmscaler.transform(np_data) # 変換

    # loc 平均
    # scale 標準偏差
    def add_noise(x, loc=0, scale=0.01):
        return abs(x + np.random.normal(loc, scale, x.shape))



    def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=1, noise=False):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(min_index + lookback, max_index, size = batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
                
            samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
            targets = np.zeros((len(rows), ))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][0]
                
            if noise==True:
                samples = add_noise(samples, loc=0, scale=0.001)
                    
            yield samples, targets


    # ハイパーパラメータ
    lookback = 100
    step = 1  # 10分おきにサンプリング
    delay = 12  # 2時間後の推移を予測する
    batch_size = 256
    #steps_per_epoch = 75
    lr = 1e-4
    recurrent_dropout1 = 0.3
    recurrent_dropout2 = 0.3
    hl1 = 512
    hl2 = 512

    train_num = 512


    train_max = np_data.shape[0] - train_num  - lookback -1
    val_max = np_data.shape[0] - 0

    # 訓練ジェネレータ
    train_gen = generator(np_data,
                            lookback=lookback,
                            delay=delay,
                            min_index=0,
                            max_index=train_max,
                            shuffle=True,
                            step=step,
                            batch_size=batch_size,
                            noise=False)

    # 検証ジェネレータ
    val_gen = generator(np_data,
                            lookback=lookback,
                            delay=delay,
                            min_index=train_max+1,
                            max_index=val_max,
                            step=step,
                            batch_size=batch_size)

    # テストジェネレータ
    test_gen = generator(np_data,
                            lookback=lookback,
                            delay=delay,
                            min_index=val_max+1,
                            max_index=None,
                            step=step,
                            batch_size=batch_size)

    # 訓練データのステップ数
    train_steps = (train_max - lookback) // batch_size

    # 検証データセット全体を調べるために　val_genから抽出する時刻刻みの数
    val_steps = (val_max - (train_max+1) - lookback) // batch_size

    # テストデータセット全体を調べるために　val_genから抽出する時刻刻みの数
    test_steps = (len(np_data) - (val_max+1) - lookback) // batch_size

    # ModelCheckpoint コールバックと EarlyStopping コールバック
    callbacks_list = [
        # 改善が止まったら訓練を中止
        keras.callbacks.EarlyStopping(
            monitor='val_loss',  # 検証データでのモデルの誤差を関し
            patience=40              # 25エポック以上に渡って誤差が改善しなければ訓練を中止
        ),
        # エポックごとに現在の重みを保存
        keras.callbacks.ModelCheckpoint(
            filepath='models4/' + number + '.h5',  # モデルの保存先となるファイルへのパス
            monitor='val_loss', 
            save_best_only=True  # 最も良いモデルを保存する
        )
    ]

    # ### LSTMのモデルの訓練と評価
    model = Sequential()
    model.add(layers.LSTM(hl1,
                        recurrent_dropout=recurrent_dropout1,
                        return_sequences=True,
                        input_shape=(None, np_data.shape[-1])))
    model.add(layers.LSTM(hl2,
                        recurrent_dropout=recurrent_dropout2,
                        return_sequences=False))
    model.add(layers.Dense(1))

    batch_input_shape = (batch_size, lookback, np_data.shape[-1])

    model = Sequential()
    model.add(layers.LSTM(hl1,
                        #recurrent_dropout=recurrent_dropout1,
                        return_sequences=True,
                        stateful=True,
                        batch_input_shape=batch_input_shape
                        ))
    model.add(layers.LSTM(hl2,
                        #recurrent_dropout=recurrent_dropout2,
                        return_sequences=False,
                        stateful=True,
                        batch_input_shape=batch_input_shape
                        ))
    model.add(layers.Dense(1))



    rmsprop = RMSprop(lr=lr)
    model.compile(optimizer=rmsprop, loss='mse')
    history = model.fit_generator(train_gen,
                                steps_per_epoch=train_steps,  # 変更
                                epochs=500,
                                callbacks=callbacks_list,
                                validation_data=val_gen,
                                validation_steps=val_steps,
                                shuffle=False)

