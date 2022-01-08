#!/usr/bin/env python
### Tensorflow keras with tensorboard and classification example
import numpy as np
import tensorflow as tf
import time


tf.random.set_seed(42)

if 0:

        from sklearn.datasets import make_classification,make_circles
        x,Y = make_classification(n_features=2,n_classes=2,n_samples=100,n_redundant=0,n_clusters_per_class=1)
        #x,Y = make_circles(n_samples=100,noise=0.03,factor=0.7)

        #### we have to show how the model enhanced with layer performs on circular data
        model = tf.keras.Sequential([
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=0.03),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )


        
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))
        # tensorboard --logdir=logs   ### will fetch the logs


        history = model.fit(x, Y, epochs=200,callbacks=[tensorboard])
        accuracy = max(history.history.get('accuracy'))*100
        # from mlxtend.plotting import plot_decision_regions
        # import matplotlib.pyplot as plt
        # plot_decision_regions(x, Y, clf=model, legend=2)
        # plt.suptitle('Accuracy'+str(accuracy))
        # plt.show()

### second example of mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def mnist_ai_model():  ## model creation
    inputs = tf.keras.Input(shape=(28*28,))
    features = tf.keras.layers.Dense(512,activation="relu")(inputs)
    features = tf.keras.layers.Dropout(0.5)(features)
    outputs = tf.keras.layers.Dense(10,activation="softmax")(features)
    model = tf.keras.Model(inputs,outputs)
    return model

ai_model = mnist_ai_model()


from tensorflow.keras.datasets import mnist
(tr_img,tr_label),(te_img,te_label)= mnist.load_data()

## normalization
tr_img = tr_img.reshape((60000,28*28)).astype("float32")/255
te_img = te_img.reshape((10000,28*28)).astype("float32")/255

tr_img, val_img = tr_img[10000:],tr_img[:10000]
tr_label, val_label = tr_label[10000:],tr_label[:10000]

plt.imshow(tr_img[1,:].reshape((28,28)),cmap='gray')
plt.show()

###add callback for storing the training session
callback_list =[
    tf.keras.callbacks.ModelCheckpoint(
        filepath = "checkpoint_ai_model",
        monitor = "val_loss",
        save_best_only = True
    ),tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))
]

        # tensorboard --logdir=logs   ### will fetch the logs
### optimizer and loss defination
ai_model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

### model training
ai_model.fit(tr_img,tr_label,epochs=5,validation_data=(val_img,val_label),callbacks=callback_list)

 #### Evaluate on unseen data
predict = ai_model.predict(te_img)
pred_label = np.argmax(predict,axis=1)
cm = confusion_matrix(te_label,pred_label)
sns.heatmap(cm,annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

 #### Evaluate on unseen data
test_eval_mat = ai_model.evaluate(te_img,te_label)  
print('Accuracy',test_eval_mat[1])