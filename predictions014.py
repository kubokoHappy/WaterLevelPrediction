import numpy as np
from matplotlib import pyplot as plt
import keras
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import time
import datetime as dt
from datetime import timedelta
import signal
import csv




# 10分おきに実行
def main():
    signal.signal(signal.SIGALRM, predict)
    signal.setitimer(signal.ITIMER_REAL, 1, 600)  # 1回目の実行までの秒数, 2回目以降の実行間隔秒数
    
    while True:
        time.sleep(1)



stay_count = 0  # 安定的な値をカウント
doun_count = 0  # 連続で下がる値をカウント
pre_tmp = 0  # 予測値の初期値

def fix(data_latest, result, stay_count_limit=10, doun_count_limit=10, stay_diff=0.0001):
    
    global stay_count
    global doun_count
    global pre_tmp
    
    final_value = result  # 最終的に出力する値
    
    if abs(result - pre_tmp) < stay_diff:  # 前の予測値と比べて差が小さかった場合
        stay_count += 1
        doun_count = 0
    elif result < pre_tmp:  # 前の予測値と比べて値が小さくなった場合
        doun_count += 1       
        stay_count = 0
    else:  # 前の予測値と比べて値が大きくなった場合
        stay_count = 0   # カウンタをリセット
        doun_count = 0  # カウンタをリセット
    pre_tmp = result  # pre_tmp を更新
    
    if  (stay_count > stay_count_limit):  # stay_count　がリミットを越した場合
        final_value = data_latest      # final_value を shift12_stack に変更(安定している場合、2時間前の値をそのまま出力)
    
    return final_value
  
model = keras.models.load_model('models4/model001.h5')



# データの予測
def predict(signum, frame):
    # データのダウンロード
    url = 'http://iotai.kd.sus.ac.jp/AIContest/download.php?from=2020/6/26%2000:00:00'
    try:
        r = requests.get(url)
        with open('./datas/all_data_to_now.csv', mode='w') as f:
            f.write(r.text)
    except requests.exceptions.RequestException as err:
        print(err)
     # データのインポート
    # data = pd.read_csv('./datas/data_24h_006.csv')
    data = pd.read_csv('./datas/all_data_to_now.csv')
    data_latest = data['zeniba'].iloc[-1]
    
    
    # 時刻を計算
    now_date = (dt.datetime.now() + dt.timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S.%f')
    last_date = dt.datetime.strptime(data.iloc[-1]['date'], '%Y-%m-%d %H:%M:%S')
    two_h = timedelta(seconds=7200)
    predicted_date = last_date + two_h
    
    edited_data = data[[
    'date', 'zeniba',
    'haregamine', 'seikaen', 'kagamiko', 'kanazawa', 'osawa', 'shiyakusho', 'izuminohoikuen', 
    'tamagawahoikuen', 'hakubutsukan', 'kohigashihoikuen', 'yonezawahoikuen', 'kashiwabarakominkan',
    'kirigamine', 'shirakabako', 'kitayatsugatake', 'okutateshina', 'minoto', 'haramura', 'toyokanko', 'tokyu',
    'alpico', 'village', 'mitsuinomori', 'kajima'
    ]]
    
    
    np_data = edited_data.iloc[:, 1:].to_numpy(dtype=float)
    mmscaler = MinMaxScaler(feature_range=(0,1)) # インスタンスの作成
    mmscaler.fit(np_data)           # 最大・最小を計算
    np_data = mmscaler.transform(np_data) # 変換
    np_data = np_data[-256:]
    np_data = np_data.reshape(1, 256, edited_data.shape[1]-1)  # 70に変更
    
    # 正規化した形を戻すためにリストの形をあわせる
    padding_cols=np.zeros(np_data.shape[2]-1)
    def padding_array(val, flag=True):
        if flag:  # 2次元の場合
            return np.array([np.insert(padding_cols,0,x)for x in val])
        else:  # 1次元の場合
            return np.array([np.insert(padding_cols,0,val)])
    
    # 予測
    result = model.predict_on_batch(np_data)
    result = mmscaler.inverse_transform(padding_array(result[0][0], flag=False))
        
    # データを補正
    final_value = fix(data_latest, result[0][0])    
    
    print(now_date, '  ', last_date, ' → ', predicted_date, '  ', final_value, '(', result[0][0], ')')
    
    # csvファイルに結果を追記
    with open('results/prediction_results014.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([now_date, last_date, predicted_date, result[0][0], final_value])

    # データの送信
    url = 'http://iotai.kd.sus.ac.jp/AIContest/pdt/pdt.php?id=romzfd0hx1&waterlevel=' + str(final_value)
    try:
        r = requests.get(url)
        print(r.url)
    except requests.exceptions.RequestException as err:
        print(err)

main()
