import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

from model import get_shallow_model
from data import DataLoader

# 학습 과정, 이상치 탐지 결과 시각화 함수
# threshold_percentile: 정상 데이터의 이상치 점수를 기준으로 임계값을 설정할 백분위수
def visualize_anomaly_detection(history, df_can_test, df_can_test_normal, df_can_test_anomaly, threshold_percentile):

    # 정상 데이터의 이상치 점수를 사용하여 임계값 설정
    threshold = np.percentile(df_can_test_normal['anomaly_score'], threshold_percentile)

    # 정상 데이터는 0, 이상치 데이터는 1로 레이블 설정
    y_true = np.concatenate([np.zeros(len(df_can_test_normal)), np.ones(len(df_can_test_anomaly))])

    # 정상 데이터와 이상치 데이터의 이상치 점수를 임계값과 비교하여 예측 레이블 설정
    # 이상치 점수가 임계값보다 크면 1(이상치), 작거나 같으면 0(정상)으로 예측
    y_pred = np.concatenate([np.where(df_can_test_normal['anomaly_score'] > threshold, 1, 0),
                            np.where(df_can_test_anomaly['anomaly_score'] > threshold, 1, 0)])

    # 이상치 분류 정확도 계산
    accuracy = np.mean(y_true == y_pred)
    '''
    EX)
    accuracy = np.mean([True, False, True, True, False])
         = np.mean([1, 0, 1, 1, 0])
         = (1 + 0 + 1 + 1 + 0) / 5
         = 0.6
    '''

    # 이상치 탐지 결과 시각화
    plt.figure(figsize=(16, 5))

    # 학습 과정 시각화
    # plt.subplot(1, 3, 1)
    # plt.plot(history.history['loss'], label='Training Loss')
    # # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # 정상 데이터와 이상치 데이터 이상치 점수 시각화
    plt.subplot(1, 3, 2)
    plt.plot(df_can_test_normal.index, df_can_test_normal['anomaly_score'], 'bo', markersize=2, label='normal')
    plt.plot(df_can_test_anomaly.index, df_can_test_anomaly['anomaly_score'], 'ro', markersize=2, label='anomaly')
    plt.axhline(y=threshold, color='g', linestyle='-', label='Threshold')
    plt.xlabel('Data Point')
    plt.ylabel('Anomaly Score')
    plt.legend()

    # anomaly_score Box Plot
    plt.subplot(1, 3, 3)
    data = [df_can_test_normal['anomaly_score'], df_can_test_anomaly['anomaly_score']]
    labels = ['normal', 'anomaly']
    plt.boxplot(data, labels=labels)
    plt.xlabel('Data Type')
    plt.ylabel('Anomaly Score')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'test.png'))
    # plt.show()

    print(f"이상치 탐지 정확도: {accuracy * 100:.2f}%")

def main(path):
    only_down_shift = True
    data_loader = DataLoader(down=only_down_shift)

    # you can check details of data
    print(
    len(data_loader.folder),     # raw folder name
    len(data_loader.anomaly),    # True or False for anomaly
    len(data_loader.dfs),        # raw data Frame from prev 0.5s to future 2s
    len(data_loader.cur_gear),   # current gear speed
    len(data_loader.tar_gear),   # target gear speed
    len(data_loader.shift),      # 'U'(up) or 'D'(down)
    )

    data_loader.split_train_test()

    # train_X = data_loader.get_train_data(16)
    # print(train_X.shape)
    test_X, test_Y = data_loader.get_test_data()
    print(test_X.shape)
    print(test_Y)

    scaler = data_loader.get_scaler()
    scaled_train_data = data_loader.get_scaled_train_data(scaler)

    scaled_test_data = data_loader.get_scaled_test_data(scaler)
    scaled_test_data0 = data_loader.get_scaled_test_data(scaler)[:len(np.where(test_Y==False)[0])]
    scaled_test_data1 = data_loader.get_scaled_test_data(scaler)[len(np.where(test_Y==False)[0]):]

    model_keys = ['AE', 'RNN-AE', 'LSTM-AE', 'GRU-AE']
    model_keys = ['LSTM-AE']
    
    adam = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    
    for model_key in model_keys :
        hiddens = 10
        features = test_X.shape[-1]
        
        checkpoint_path = os.path.join("training/{}_{}".format(model_key, features), "cp.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1, 
                                                        monitor='loss', 
                                                        save_best_only=True)
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

        model = get_shallow_model(model_key, 250, features, hiddens, 2)

        model.compile(optimizer=adam, loss='mse')
        history = model.fit(x=scaled_train_data, y=scaled_train_data, 
                            epochs=500, shuffle=True, batch_size=32, callbacks = [cp_callback, es_callback])
        ###########################################################################
        # model.load_weights(checkpoint_path).expect_partial()
        
        # 이상치 점수 계산
        reconstructions = model.predict(scaled_test_data)
        squared_error = np.square(scaled_test_data - reconstructions)
        mse = np.mean(squared_error, axis=(1, 2))
        df_can_test = pd.DataFrame()
        df_can_test['anomaly_score'] = mse

        # 정상 데이터와 이상치 데이터 분리
        df_can_test_normal = df_can_test[:len(np.where(test_Y==False)[0])]
        df_can_test_anomaly = df_can_test[len(np.where(test_Y==False)[0]):]

        # 시각화
        # 복원 오류가 정상 데이터에서 얻은 임계치를 초과하는 경우 이상치로 분류
        visualize_anomaly_detection('history', df_can_test, df_can_test_normal, df_can_test_anomaly, threshold_percentile=80)
        ###########################################################################
        
        reconstructions = model.predict(scaled_test_data)
        test_error = tf.keras.losses.mae(reconstructions, scaled_test_data)

        reconstructions0 = model.predict(scaled_test_data0)
        test_error0 = tf.keras.losses.mae(reconstructions0, scaled_test_data0)

        reconstructions1 = model.predict(scaled_test_data1)
        test_error1 = tf.keras.losses.mae(reconstructions1, scaled_test_data1)

        auroc = roc_auc_score(test_Y, tf.reduce_mean(test_error, axis=-1))
        print("AUROC = {}".format(auroc))
        
        plt.hist(tf.reduce_mean(test_error0, axis=-1), bins=50)
        plt.hist(tf.reduce_mean(test_error1, axis=-1), bins=50)
        plt.xlabel("Test error")
        plt.ylabel("# of examples")
        plt.legend(['GT False', 'GT True'], loc='upper right')
        plt.savefig(os.path.join(path, '{}_{}_{}_{:.3f}.png'.format(model_key, features, only_down_shift, auroc)))
        plt.clf()
        # plt.show()
        
if __name__ == '__main__':
    base_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(base_path, 'fig')
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    main(path)
