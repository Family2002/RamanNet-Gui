"""
   train the RamanNet model
"""

from codes.data_processing import segment_spectrum_batch
from codes.RamanNet_model import RamanNet
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
import tensorflow as tf

def train_model(X, y, w_len, dw, epochs, val_split, model_path, plot=True, progress_callback=None):
    # 数据分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42,stratify=y)

    # 将标签转换为 one-hot 编码
    num_classes = len(np.unique(y))
    Y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    Y_val_onehot = to_categorical(y_val, num_classes=num_classes)

    # 数据预处理
    X_train = segment_spectrum_batch(X_train, w_len, dw)
    X_val = segment_spectrum_batch(X_val, w_len, dw)

    mdl = RamanNet(X_train[0].shape[1], len(X_train), num_classes)

    losses = {
        "embedding": "mse",
        "classification": CategoricalCrossentropy(),
    }
    lossWeights = {"embedding": 0.5, "classification": 0.5}

    mdl.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=[CategoricalAccuracy()])

    checkpoint_ = ModelCheckpoint(model_path, verbose=1, monitor='val_loss', save_best_only=True, mode='min')  
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.00000001, verbose=1)

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch, epochs)

    callbacks = [checkpoint_, reduce_lr, ProgressCallback()]

    training_history = mdl.fit(x=X_train, y=[y_train, Y_train_onehot], batch_size=256, epochs=epochs, 
                               validation_data=(X_val, [y_val, Y_val_onehot]), 
                               callbacks=callbacks, verbose=1)    

    if plot:
        # 绘图代码保持不变
        pass

    return mdl, training_history
