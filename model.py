# Script to train a semi-supervised generative model
# with a variational autoencoder 
#
# All rights reserved to Amar Shah

import numpy as np 
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, merge
from keras.objectives import binary_crossentropy as bce
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization as BN
from keras.datasets import mnist
from keras.optimizers import Adam

### parameters #####################################################
n_batch = 100
n_epoch = 2
a_dim = 100
z_dim = 100
img_dim = 728
nb_classes = 10
kl_weight = K.variable(0.)
n_label_per_class = 10

### data loading #######################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape and normalise data to probabilities
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# choose labelled / unlabelled split
x_label = []
y_label = []
x_unlabel = []
for i in xrange(nb_classes):
	X_i = X_train[Y_train==i, :]
	np.random.shuffle(X_i)
	x_label = np.concatenate([x_label, X_i[:n_label_per_class, :]])
	y_label = np.concatenate([y_label, i*np.ones((n_label_per_class,))])
	x_unlabel = np.concatenate([x_label, X_i[n_label_per_class:, :]])

# shuffle data before training 
inds = range(n_label_per_class * nb_classes)
np.random.shuffle(inds)
x_label = x_label[inds, :]
y_label = y_label[inds, :]
np.random.shuffle(x_unlabel)


### define encoders and decoders #######################################################
# first a data sampling layer
data_sampler = Lambda(lambda arg : K.random_binomial(arg.shape, arg),
	output_shape=(img_dim,))

# encoder for a, the auxiliary latent variable
a_l1_en = Dense(500, activation="relu")
a_l2_en = BN()
a_l3_en = Dense(500, activation="relu")
a_l4_en = BN()
a_l5a_en = Dense(a_dim)
a_l5b_en = Dense(a_dim)

def encode_a(x):
	h = a_l4_en(a_l3_en(a_l2_en(a_l1_en(x))))
	a_mean_en = a_l5a_en(h)
	a_logvar_en = a_l5b_en(h)
	return a_mean_en, a_logvar_en

# encoder for y, the label
y_l1_en = Dense(500, activation="relu")
y_l2_en = BN()
y_l3_en = Dense(500, activation="relu")
y_l4_en = BN()
y_l5_en = Dense(nb_classes, activation="softmax")

def encode_y(a, x):
	merged = merge([a, x], mode="concat", concat_axis=-1)
	y_prob_en = y_l5_en(y_l4_en(y_l3_en(y_l2_en(y_l1_en(merged)))))
	return y_prob_en

# encoder for z, the main latent variable
z_l1_en = Dense(500, activation="relu")
z_l2_en = BN()
z_l3_en = Dense(500, activation="relu")
z_l4_en = BN()
z_l5a_en = Dense(z_dim)
z_l5b_en = Dense(z_dim)

def encode_z(a, y, x):
	merged = merge([a, y, x], mode="concat", concat_axis=-1)
	h = z_l4_en(z_l3_en(z_l2_en(z_l1_en(merged))))
	z_mean_en = z_l5a_en(h)
	z_logvar_en = z_l5b_en(h)
	return z_mean_en, z_logvar_en

# decoder for a
a_layer1_de = Dense(500, activation="relu")
a_layer2_de = BN()
a_layer3_de = Dense(500, activation="relu")
a_layer4_de = BN()
a_layer5a_de = Dense(a_dim)
a_layer5b_de = Dense(a_dim)

def decode_a(y, z):
	merged = merge([y, z], mode="concat", concat_axis=-1)
	h = a_l4_de(a_l3_de(a_l2_de(a_l1_de(merged))))
	a_mean_de = a_l5a_de(h)
	a_logvar_de = a_l5b_de(h)
	return a_mean_de, a_logvar_de

# decoder for x
x_l1_de = Dense(500, activation="relu")
x_l2_de = BN()
x_l3_de = Dense(500, activation="relu")
x_l4_de = BN()
x_l5_de = Dense(img_dim, activation="softmax")

def decode_x(a, y, z):
	merged = merge([a, y, z], mode="concat", concat_axis=-1)
	x_prob_de = x_l5_de(x_l4_de(x_l3_de(x_l2_de(x_l1_de(merged)))))
	return x_prob_de

### define some useful functions going forward ################################

## BE CAREFUL ABOUT DIFFERENT BATCH SIZES
def sampling(args):
    lat_mean, lat_logvar, lat_dim = args
    eps = K.random_normal(shape=(n_batch, lat_dim), mean=0., std=0.1)
    return lat_mean + K.exp(lat_logvar / 2) * eps

def mvn_kl(mean1, logvar1, mean2=None, logvar2=None):
	# computes kl between N(mean1, var1) and N(mean2, var2)
	# if mean2 is None, assumes mean2=logvar2=0
	if mean2 is None:
		mean2 = K.zeros(shape=mean1.shape)
		logvar2 = K.zeros(shape=logvar1.shape)

	kl = 0.5 * K.sum(logvar2 - logvar1 - 1 + \
		(K.exp(logvar1) + (mean1 - mean2) ** 2) / K.exp(logvar2), axis=-1)
	return kl


### define model for labelled data #################################################################
x = Input(shape=(img_dim,))
y = Input(shape=(1,))

# sample bernoullis for input
x_s = data_sampler(x)

# encode and sample a
a_mean_en, a_logvar_en = encode_a(x)
a = Lambda(sampling, output_shape=(a_dim,))([a_mean_en, a_logvar_en])

# encode y
y_prob_en = encode_y(a, x):

# encode and sample z
z_mean_en, z_logvar_en = encode_z(a, y, x)
z = Lambda(sampling, output_shape=(z_dim,))([z_mean_en, z_logvar_en])

# decode a
a_mean_de, a_logvar_de = decode_a(y, z)

# decode x
x_prob_de = decode_x(a, y, z)

# define vae loss for labelled data 
def vae_label_loss(x, x_prob_de):
	x_loss = img_dim * bce(x, x_prob_de)
	y_loss = - K.log(nb_classes) 
	kl_a_loss = mvn_kl(a_mean_en, a_logvar_en, a_mean_de, a_logvar_de)
	kl_z_loss = mvn_kl(z_mean_en, z_logvar_en)

	rec_loss = x_loss + y_loss
	kl_loss = kl_a_loss + kl_z_loss
	total_loss = rec_loss + kl_weight * kl_loss

	return total_loss 


### define model for unlabelled data ###########################################################
x_u = Input(shape=(img_dim,))

# sample bernoullis for input
x_u_s = data_sampler(x_u)

# encode and sample a_u
a_u_mean_en, a_u_logvar_en = encode_a(x_u)
a_u = Lambda(sampling, output_shape=(a_dim,))([a_u_mean_en, a_u_logvar_en])

# encode y_u
y_u_prob_en = encode_y(a_u, x_u)

# encode z, sample z, decode a and x for each possible y
z_u_mean_en = z_u_logvar_en = []
a_u_mean_de = a_u_logvar_de = []
x_u_prob_de = []
for i in range(nb_classes):
	y_i = K.ones(shape=(len(x_u), 1)) + i
	# encode z_u
	z_u_mean_en[i], z_u_logvar_en[i] = encode_z(a_u, y_i, x_u)
	# sample z_u
	z_u = Lambda(sampling, output_shape=(z_dim,))([z_u_mean_en, z_u_logvar_en])

	# decode a_u
	a_u_mean_de[i], a_u_logvar_de[i] = decode_a(y_i, z_u)

	# decode x_u
	x_u_prob_de[i] = decode_x(a_u, y_i, z_u)


# define vae loss for labelled data 
def vae_unlabel_loss(x_u, x_u_prob_de):
	for i in xrange(nb_classes):
		y_i_prob = y_u_probs_en[:, i]
		x_u_prob_de_y = x_u_prob_de[i]
		a_u_mean_de_y = a_u_mean_de_y[i] 
		a_u_logvar_de_y = a_u_logvar_de_y[i] 
		z_u_mean_en_y = z_u_mean_en_y[i] 
		z_u_logvar_en_y = z_u_logvar_en_y[i] 

		x_u_loss = img_dim * bce(x_u, x_u_prob_de_y)
		kl_y_loss = K.log(nb_classes * y_i_prob)
		kl_a_u_loss = mvn_kl(a_u_mean_en, a_u_logvar_en,
			a_u_mean_de_y, a_u_logvar_de_y)
		kl_z_u_loss = mvn_kl(z_u_mean_en_y, z_u_logvar_en_y)

		if i==0:
			rec_loss = y_i_prob * x_u_loss
			kl_loss = y_i_prob * (kl_y_loss + kl_a_u_loss + kl_z_u_loss)
		else:
			rec_loss += y_i_prob * x_u_loss
			kl_loss += y_i_prob * (kl_y_loss + kl_a_u_loss + kl_z_u_loss)

	total_loss = rec_loss + kl_weight * kl_loss		
	return total_loss

### define and compile model ###############################################################################
ss_model = Model(input=[x, y, x_u], output=[x_prob_de, x_u_prob_de])

optimizer = Adam(lr=3e-4)
ss_model.compile(optimizer=optimizer,
	loss={'x_prob_de': 'vae_label_loss', 'x_u_prob_de': 'vae_unlabel_loss'},
	loss_weights={'x_prob_de': 0.5, 'x_u_prob_de': 0.5})

class MyCallback(Callback):
	def __init__(self, kl_weight):
		super(MyCallback, self).__init__()
		self.kl_weight = kl_weight

	def on_epoch_end(self, epoch):
		new_value = np.min([epoch * 0.005, 1.0])
		self.kl_weight.set_value(new_value)

cb = MyCallback(kl_weight)

ss_model.fit([x_label, y_label, x_unlabel],
	[x_label, y_label, x_unlabel],
	shuffle=True, batch_size=n_batch, nb_epoch=n_epoch, callbacks=cb)

import pdb
pdb.set_trace()
