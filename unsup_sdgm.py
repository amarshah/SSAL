'''
Attempt to replicate unsupervised learning results on MNIST from
"Auxiliary Deep Generative Models" https://arxiv.org/abs/1602.05473
'''
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Input, Dense, Lambda, merge
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
from keras.callbacks import Callback
from keras.objectives import binary_crossentropy as bce
from keras.optimizers import Adam
from keras.datasets import mnist

batch_size = 100
img_dim = 784
z_dim = 100
a_dim = 100
intermediate_dim = 500
nb_epoch = 400
kl_weight = K.variable(0.)

def mvn_kl(mean1, logvar1, mean2=None, logvar2=None):
    # computes kl between N(mean1, var1) and N(mean2, var2)
    # if mean2 is None, assumes mean2=logvar2=0
    if mean2 is None:
        mean2 = K.zeros_like(mean1)
        logvar2 = K.zeros_like(logvar1)

    kl = 0.5 * K.sum(logvar2 - logvar1 - 1 + \
        (K.exp(logvar1) + (mean1 - mean2) ** 2) / K.exp(logvar2), axis=-1)

    return kl

x = Input(batch_shape=(batch_size, img_dim))

x_s = Lambda(lambda arg : K.random_binomial(arg.shape, arg),
    output_shape=(img_dim,))(x)

# encode
h1 = BN()(Dense(intermediate_dim, activation='relu')(x_s))
h3 = BN()(Dense(intermediate_dim, activation='relu')(h1))
a_mean_en = Dense(z_dim)(h3)
a_logvar_en = Dense(z_dim)(h3)
def sampling_a(args):
    a_mean, a_log_var = args
    epsilon = K.random_normal(shape=(batch_size, a_dim))
    return a_mean + K.exp(a_log_var / 2) * epsilon

a = Lambda(sampling_a, output_shape=(a_dim,))([a_mean_en, a_logvar_en])

merged = merge([a, x_s], mode="concat", concat_axis=-1)
h4 = BN()(Dense(intermediate_dim, activation='relu')(merged))
h6 = BN()(Dense(intermediate_dim, activation='relu')(h4))
z_mean_en = Dense(z_dim)(h6)
z_logvar_en = Dense(z_dim)(h6)
def sampling_z(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, z_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling_z, output_shape=(z_dim,))([z_mean_en, z_logvar_en])

# decode
g1 = BN()(Dense(intermediate_dim, activation='relu')(z))
g3 = BN()(Dense(intermediate_dim, activation='relu')(g1))
a_mean_de = Dense(a_dim)(g3)
a_logvar_de = Dense(a_dim)(g3)

merged = merge([z, a], mode="concat", concat_axis=-1)
g4 = BN()(Dense(intermediate_dim, activation='relu')(merged))
g6 = BN()(Dense(intermediate_dim, activation='relu')(g4))
x_mean = Dense(img_dim, activation='sigmoid')(g6)

# compute loss
def vae_loss(x, x_mean):
    xent_loss = img_dim * bce(x_s, x_mean)

    kl_z_loss = mvn_kl(z_mean_en, z_logvar_en)
    kl_a_loss = mvn_kl(a_mean_en, a_logvar_en, a_mean_de, a_logvar_de)
    kl_loss = kl_z_loss + kl_a_loss

    # return kl_loss + xent_loss
    return kl_weight * kl_loss + xent_loss

vae = Model(x, x_mean)
optimizer = Adam(lr=2e-4)
vae.compile(optimizer=optimizer, loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

class MyCallback(Callback):
    def __init__(self, kl_weight):
        super(MyCallback, self).__init__()
        self.kl_weight = kl_weight

    def on_epoch_end(self, epoch, logs={}):
        new_value = np.min([epoch * 0.005, 1.0]).astype("float32")
        self.kl_weight.set_value(new_value)
        print kl_weight.get_value()

cb = MyCallback(kl_weight)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        callbacks=[cb],
        validation_data=(x_test, x_test))

