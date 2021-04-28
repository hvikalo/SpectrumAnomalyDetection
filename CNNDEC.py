import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= '1'
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np
from sklearn.cluster import KMeans
import h5py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    
    return (weight.T / weight.sum(1)).T

def split_data(X, Y):
    index = list(range(X.shape[0]))
    np.random.seed(2020)
    np.random.shuffle(index)

    train_proportion = 0.7
    validation_proportion = 0.25
    test_proportion = 0.25

    X_train = X[index[:int(X.shape[0] * train_proportion)], :48, :48]
    Y_train = Y[index[:int(X.shape[0] * train_proportion)]]
    X_validation = X[index[int(X.shape[0] * train_proportion) : int(X.shape[0] * (train_proportion + validation_proportion))], :48, :48]
    Y_validation = Y[index[int(X.shape[0] * train_proportion) : int(X.shape[0] * (train_proportion + validation_proportion))]]
    X_test = X[index[int(X.shape[0] * (train_proportion + validation_proportion)):], :48, :48]
    Y_test = Y[index[int(X.shape[0] * (train_proportion + validation_proportion)):]]
    
    for i in range(X_train.shape[0]):
        data_min = np.nanmin(X_train[i, :, :])
        data_max = np.nanmax(X_train[i, :, :])
        new_max, new_min = 1, 0
        X_train[i, :, :] = (X_train[i, :, :] - data_min) * (new_max - new_min) / (data_max - data_min) + new_min
        
    for i in range(X_validation.shape[0]):
        data_min = np.nanmin(X_validation[i, :, :])
        data_max = np.nanmax(X_validation[i, :, :])
        new_max, new_min = 1, 0
        X_validation[i, :, :] = (X_validation[i, :, :] - data_min) * (new_max - new_min) / (data_max - data_min) + new_min
        
    for i in range(X_test.shape[0]):
        data_min = np.nanmin(X_test[i, :, :])
        data_max = np.nanmax(X_test[i, :, :])
        new_max, new_min = 1, 0
        X_test[i, :, :] = (X_test[i, :, :] - data_min) * (new_max - new_min) / (data_max - data_min) + new_min
    
    return X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)), Y_train.astype(int), X_validation.reshape((X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)), Y_validation.astype(int), X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)), Y_test.astype(int)

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights = None, alpha = 1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim = 2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = int(input_shape[1])
        self.input_spec = InputSpec(dtype = K.floatx(), shape = (None, input_dim))
        self.clusters = self.add_weight(shape = (self.n_clusters, input_dim), initializer = 'glorot_uniform', name = 'clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1 / (1 + dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis = 1) - self.clusters), axis = 2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis = 1))
        
        return q

# load data
with h5py.File('anomaly_50.h5', 'r') as hf:
    X = hf['anomaly_50'][:]
    
with h5py.File('label_50.h5', 'r') as hf:
    Y = hf['label_50'][:]   
    
X_train, Y_train, X_validation, Y_validation, X_test, Y_test = split_data(X, Y)

# hyper-parameters
n_clusters = 2
lam = 0.1
learning_rate = 10 ** -3
batch_size = 256
n_pretrain_epoch = 100
itermax = int(2e4)
tol = 10 ** -3
update_interval = 140
len_code = 16
drop_prob = 0

# convolutional auto-encoder
Input = tf.keras.Input(shape = (X_train.shape[1], X_train.shape[2], 1),
                       name = 'Input')
# convolutional layer - 1
conv_1 = tf.keras.layers.Conv2D(filters = 32,
                                kernel_size = [5, 5],
                                strides = (2, 2),
                                padding = 'same',
                                name = 'conv_1')(Input)
PRelu_1 = tf.keras.layers.PReLU(name = 'PRelu_1')(conv_1)
drop_1 = tf.keras.layers.Dropout(drop_prob, name = 'drop_1')(PRelu_1)

conv_2 = tf.keras.layers.Conv2D(filters = 64,
                                kernel_size = [5, 5],
                                strides = (2, 2),
                                padding = 'same',
                                name = 'conv_2')(drop_1)
PRelu_2 = tf.keras.layers.PReLU(name = 'PRelu_2')(conv_2)
drop_2 = tf.keras.layers.Dropout(drop_prob, name = 'drop_2')(PRelu_2)

conv_3 = tf.keras.layers.Conv2D(filters = 128,
                                kernel_size = [3, 3],
                                strides = (2, 2),
                                padding = 'same',
                                name = 'conv_3')(drop_2)
PRelu_3 = tf.keras.layers.PReLU(name = 'PRelu_3')(conv_3)
drop_3 = tf.keras.layers.Dropout(drop_prob, name = 'drop_3')(PRelu_3)
# flatten
flatten = tf.keras.layers.Flatten(name = 'flatten')(drop_3)
# code
code = tf.keras.layers.Dense(units = len_code,
                             name = 'code')(flatten)
# dense
dense_1 = tf.keras.layers.Dense(units = flatten.shape[1],
                                name = 'dense_1')(code)
PRelu_4 = tf.keras.layers.PReLU(name = 'PRelu_4')(dense_1)
# reshape
reshape = tf.keras.layers.Reshape((drop_3.shape[1], drop_3.shape[2], drop_3.shape[3]), 
                                   name = 'reshape')(PRelu_4)
# transposed convolution layer
convT_1 = tf.keras.layers.Conv2DTranspose(filters = 64,
                                          kernel_size = [3, 3],
                                          strides = (2, 2),
                                          padding = 'same',
                                          name = 'convT_1')(reshape)
PRelu_5 = tf.keras.layers.PReLU(name = 'PRelu_5')(convT_1)
drop_4 = tf.keras.layers.Dropout(drop_prob, name = 'drop_4')(PRelu_5)

convT_2 = tf.keras.layers.Conv2DTranspose(filters = 32,
                                          kernel_size = [5, 5],
                                          strides = (2, 2),
                                          padding = 'same',
                                          name = 'convT_2')(drop_4)
PRelu_6 = tf.keras.layers.PReLU(name = 'PRelu_6')(convT_2)
drop_5 = tf.keras.layers.Dropout(drop_prob, name = 'drop_5')(PRelu_6)

convT_3 = tf.keras.layers.Conv2DTranspose(filters = 1,
                                          kernel_size = [5, 5],
                                          strides = (2, 2),
                                          padding = 'same',
                                          name = 'convT_3')(drop_5)
PRelu_7 = tf.keras.layers.PReLU(name = 'PRelu_7')(convT_3)
drop_6 = tf.keras.layers.Dropout(drop_prob, name = 'drop_6')(PRelu_7)

# clustering layer
clustering_layer = ClusteringLayer(n_clusters, name = 'clustering_layer')(code)

model = tf.keras.Model(inputs = Input,
                       outputs = [drop_6, clustering_layer])
model.summary()

model.compile(loss = ['mean_squared_error', 'kld'],
              loss_weights = [1 - lam, lam],
              optimizer = tf.keras.optimizers.Adam(lr = learning_rate))

# pretrain
pretrain = tf.keras.Model(Input, drop_6)
pretrain.compile(loss = 'mean_squared_error',
                 optimizer = tf.keras.optimizers.Adam(lr = learning_rate))

pretrain_history = pretrain.fit(x = X_train,
                                y = X_train,
                                batch_size = batch_size,
                                validation_data = (X_validation, X_validation),
                                epochs = n_pretrain_epoch)

# initialize cluster centers using k-means
encoder = tf.keras.Model(Input, code)
kmeans = KMeans(n_clusters = n_clusters, n_init = 20)
y_pred = kmeans.fit_predict(encoder.predict(X_train))
y_pred_last = np.copy(y_pred)
model.get_layer(name = 'clustering_layer').set_weights([kmeans.cluster_centers_])

print(acc(Y_train, y_pred))

plot_loss = [[], [], []]
index = 0

for i in range(itermax):
    if i % update_interval == 0:
        _, q = model.predict(X_train)
        p = target_distribution(q)
        y_pred = q.argmax(1)
        try:
            y_pred_last
        except:
            y_pred_last = np.copy(y_pred)
            
        if i > 0:
            print('iteration: {}, acc: {}, reconstruction loss: {}, loss: {}, clustering loss: {}'.format(i, acc(Y_train, y_pred), loss[0], loss[1], loss[2]))
        
        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if i > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stop training.')
            print(i)
            break
    
    if (index + 1) * batch_size > X_train.shape[0]:
        loss = model.train_on_batch(x = X_train[index * batch_size::],
                                    y = [X_train[index * batch_size::], p[index * batch_size::]])
        index = 0
    else:
        loss = model.train_on_batch(x = X_train[index * batch_size:(index + 1) * batch_size],
                                    y = [X_train[index * batch_size:(index + 1) * batch_size],
                                         p[index * batch_size:(index + 1) * batch_size]])
        index += 1
    
    for j in range(3):
        plot_loss[j].append(loss[j])

_, prob_validation = model.predict(X_validation)
y_validation = prob_validation.argmax(1)
print(acc(Y_validation, y_validation))

_, prob_test = model.predict(X_test)
y_test = prob_test.argmax(1)
print(acc(Y_test, y_test))
