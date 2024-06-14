from gc import callbacks
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

    model_keys = ['AE', 'RNN-AE', 'LSTM-AE', 'GRU-AE']
    model_keys = ['LSTM-AE', 'GRU-AE']
    
    adam = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    
    for model_key in model_keys :
        checkpoint_path = os.path.join("training/{}".format(model_key), "cp.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1, 
                                                        monitor='loss', 
                                                        save_best_only=True)
        hiddens = 10
        features = 4
        model = get_shallow_model(model_key, 250, features, hiddens, 2)

        model.compile(optimizer=adam, loss='mse')
        history = model.fit(x=scaled_train_data, y=scaled_train_data, 
                            epochs=500, shuffle=True, batch_size=32, callbacks = [cp_callback])
        ###########################################################################
        scaled_test_data = data_loader.get_scaled_test_data(scaler)
        scaled_test_data0 = data_loader.get_scaled_test_data(scaler)[:len(np.where(test_Y==False)[0])]
        scaled_test_data1 = data_loader.get_scaled_test_data(scaler)[len(np.where(test_Y==False)[0]):]

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
