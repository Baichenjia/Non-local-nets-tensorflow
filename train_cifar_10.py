# -*- coding: utf-8 -*- 
import os
import tensorflow as tf
import numpy as np
from non_local_tf import non_local_block
tf.enable_eager_execution()

n_class = 100

layers = tf.keras.layers
class CNN(tf.keras.Model):
    def __init__(self, non_local=True):
        super(CNN, self).__init__()
        self.non_local = non_local
        self.conv1 = layers.Conv2D(16, (3,3), padding='same')
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(32, (3,3), padding='same')
        self.pool2 = layers.MaxPool2D(2)
        self.conv3 = layers.Conv2D(64, (3,3), padding='same')
        self.pool3 = layers.MaxPool2D(2)
        self.conv4 = layers.Conv2D(128, (3,3), padding='same')
        self.conv5 = layers.Conv2D(128, (3,3), padding='same')
        self.avgPool = layers.AveragePooling2D((4,4))
        self.dense = layers.Dense(n_class)

        # non-local
        if non_local == True:
            self.atten1 = non_local_block(input_channel=128, input_shape=4, mode='embedded')
            self.atten2 = non_local_block(input_channel=128, input_shape=4, mode='embedded')

    def predict(self, x):
        x = self.conv1(x)  # (batchsize, 32, 32, 16)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)  # (batchsize, 16, 16, 32)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)   # (batchsize, 8, 8, 64)
        x = tf.nn.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)    # (batchsize, 4, 4, 128)
        x = tf.nn.relu(x)
        [x, w1] = self.atten1(x) if self.non_local else [x, None]
        x = self.conv5(x)    # (batchsize, 4, 4, 128)
        x = tf.nn.relu(x)
        [x, w2] = self.atten2(x) if self.non_local else [x, None]                  
        x = self.avgPool(x)  # (batchsize, 1, 1, 128)
        x = tf.squeeze(x)    # (batchsize, 128)

        logits = self.dense(x)   # (batchsize, 100)
        return logits, [w1,w2] 

    def loss_fn(self, X, y):
        preds, _ = self.predict(X)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=preds)
        return loss 

    def acc_fn(self, X, y):
        preds = self.predict(X)[0].numpy()
        acc = np.sum(np.argmax(preds, axis=1) == y.numpy(), dtype=np.float32) / X.numpy().shape[0]
        return acc 


def train(model, dataset, test_data, test_labels, optimizer, epoches, logdir):
    train_losses, val_losses, val_accs = [], [], []
    test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)
    best_acc = 0.
    # train
    for epoch in range(epoches):
        losses = []
        for (batch, (inp, targ)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = model.loss_fn(inp, targ)
            gradients = tape.gradient(loss, model.trainable_variables)
            # print("loss: ", loss.numpy(), ",\tacc: ", model.acc_fn(inp, targ)*100, "%")
            optimizer.apply_gradients(zip(gradients, model.variables))
            losses.append(loss)
        
        print("Epoch :", epoch, ", train loss :", np.mean(losses))
        val_loss = model.loss_fn(test_data, test_labels)
        val_acc = model.acc_fn(test_data, test_labels)
        print("Epoch :", epoch, ", val loss", val_loss.numpy(),", valid acc:", val_acc*100, "%")
        train_losses.append(np.mean(losses))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_acc > best_acc:
            # model.save_weights(os.path.join(logdir, "model_best_"+str(val_acc)+".h5"))
            model.save_weights(os.path.join(logdir, "model_best.h5"))
            best_acc = val_acc
    return np.array(train_losses), np.array(val_losses), np.array(val_accs)


if __name__ == '__main__':
    # dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.
    y_train = y_train.reshape(y_train.shape[0]).astype(np.int)
    y_test = y_test.reshape(y_test.shape[0]).astype(np.int)

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(50000)
    dataset = dataset.batch(64, drop_remainder=True)

    # model 
    non_local = True
    learning_rate = tf.Variable(1e-3, name="learning_rate")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model = CNN(non_local=non_local)

    # dir
    logdir = "log-non-local" if non_local else "log"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # train
    loss1, loss2, accs = train(model, dataset, X_test, y_test, optimizer, epoches=30, logdir=logdir)

    # plot
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(loss1.shape[0]), loss1)
    plt.plot(np.arange(loss2.shape[0]), loss2)
    plt.legend(["train loss", "val loss"])
    plt.grid()
    plt.savefig(os.path.join(logdir, "loss.png"))
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(accs.shape[0]), accs)
    plt.legend(["val acc"])
    plt.grid()
    plt.savefig(os.path.join(logdir, "acc.png"))
    plt.close()
        