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

save_folder = '/scratch/as793/ssal/'
batch_size = 100
img_dim = 784
z_dim = 100
a_dim = 100
intermediate_dim = 500
nb_epoch = 1#400
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

### encoder ##################################################
# encode a
h1 = Dense(intermediate_dim, activation='relu')
h2 = BN(mode=2)
h3 = Dense(intermediate_dim, activation='relu')
h4 = BN(mode=2)
h5a = Dense(a_dim)
h5b = Dense(a_dim)

tmp = h4(h3(h2(h1(x_s))))
a_mean_en = h5a(tmp)
a_logvar_en = h5b(tmp)
def sampling_a(args):
    a_mean, a_log_var = args
    epsilon = K.random_normal(shape=(batch_size, a_dim))
    return a_mean + K.exp(a_log_var / 2) * epsilon

a = Lambda(sampling_a, output_shape=(a_dim,))([a_mean_en, a_logvar_en])

# encode z
j1 = Dense(intermediate_dim, activation='relu')
j2 = BN(mode=2)
j3 = Dense(intermediate_dim, activation='relu')
j4 = BN(mode=2)
j5a = Dense(z_dim)
j5b = Dense(z_dim)

merged = merge([a, x_s], mode="concat", concat_axis=-1)
tmp = j4(j3(j2(j1(merged))))
z_mean_en = j5a(tmp)
z_logvar_en = j5b(tmp)
def sampling_z(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, z_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling_z, output_shape=(z_dim,))([z_mean_en, z_logvar_en])

### decoder #############################################################
# decode x
g1 = Dense(intermediate_dim, activation='relu')
g2 = BN(mode=2)
g3 = Dense(intermediate_dim, activation='relu')
g4 = BN(mode=2)
g5 = Dense(intermediate_dim, activation='softmax')

x_mean = g5(g4(g3(g2(g1(z)))))

# decode a
k1 = Dense(intermediate_dim, activation='relu')
k2 = BN(mode=2)
k3 = Dense(intermediate_dim, activation='relu')
k4 = BN(mode=2)
k5a = Dense(a_dim)
k5b = Dense(a_dim)

merged = merge([z, x_s], mode="concat", concat_axis=-1)
tmp = k4(k3(k2(k1(merged))))
a_mean_de = k5a(tmp)
a_logvar_de = k5b(tmp)

### compute loss and make model ########################################
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

### make other models for debugging #####################################

a_encoder = Model(x, [a_mean_en, a_logvar_en])

a_in = Input(shape=(a_dim,))
merged = merge([a_in, x_s], mode="concat", concat_axis=-1)
tmp = j4(j3(j2(j1(merged))))
z_mean_en_out = j5a(tmp)
z_logvar_en_out = j5b(tmp)
z_encoder = Model([x, a_in], [z_mean_en_out, z_logvar_en_out])

z_in = Input(shape=(z_dim,))
merged = merge([z_in, x_s], mode="concat", concat_axis=-1)
tmp = k4(k3(k2(k1(merged))))
a_mean_de_out = k5a(tmp)
a_logvar_de_out = k5b(tmp)
a_decoder = Model([x, z_in], [a_mean_de_out, a_logvar_de_out])

# train the VAE on MNIST digits #########################################
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

vae.save(save_folder + 'vae.h5')
a_encoder.save(save_folder + 'a_encoder.h5')
z_encoder.save(save_folder + 'z_encoder.h5')
a_decoder.save(save_folder + 'a_decoder.h5')

