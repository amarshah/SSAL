# Script to train a semi-supervised generative model
# with a variational autoencoder 
#
# All rights reserved to Amar Shah


import numpy as np 

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, merge
from keras.objectives import binary_crossentropy as bce

a_dim = 100
z_dim = 100
img_dim = 728
nb_classes = 10

# define model for labelled data #################################
x = Input(shape=(img_dim,))
y = Input(shape=(1,))

def sampling(args):
    lat_mean, lat_logvar, lat_dim = args
    eps = K.random_normal(shape=(n_batch, lat_dim), mean=0., std=0.1)
    return lat_mean + K.exp(lat_logvar / 2) * eps

# encoder for a, the auxiliary latent variable
a_layer1_en = Dense(500, activation="relu")(x)
a_layer2_en = Dense(500, activation="relu")(a_layer1_en)
a_mean_en = Dense(a_dim)(a_layer2_en)
a_logvar_en = Dense(a_dim)(a_layer2_en)
# sample a
a = Lambda(sampling, output_shape=(a_dim,))([a_mean_en, a_logvar_en])

# encoder for z, the main latent variable
merged_input = merge([a, y, x], mode="concat", concat_axis=-1)
z_layer1_en = Dense(500, activation="relu")(merged_input)
z_layer2_en = Dense(500, activation="relu")(z_layer1_em)
z_mean_en = Dense(z_dim)(z_layer2_en)
z_logvar_en = Dense(z_dim)(z_layer2_en)
# sample z
z = Lambda(sampling, output_shape=(z_dim,))([z_mean_en, z_logvar_en])

# decoder for a
merged_input = merge([y, x], mode="concat", concat_axis=-1)
a_layer3_de = Dense(500, activation="relu")(merged_input)
a_layer4_de = Dense(500, activation="relu")(a_layer3_de)
a_mean_de = Dense(a_dim)(a_layer4_de)
a_logvar_de = Dense(a_dim)(a_layer4_de)

# decoder for x
merged_input = merge([a, y, z], mode="concat", concat_axis=-1)
x_layer1_de = Dense(500, activation="relu")(merged_input)
x_layer2_de = Dense(500, activation="relu")(x_layer1_de)
x_prob_de = Dense(img_dim, activation="softmax")(x_layer2_de)

# define vae loss for labelled data 
def vae_label_loss(x, x_prob_de):
	x_loss = img_dim * bce(x, x_prob_de)
	y_loss = - K.log(nb_classes) 
	kl_a_loss = 0.5 * K.sum(a_logvar_de - a_logvar_en - 1 \
		+ K.exp(a_logvar_en - a_logvar_de) \
		+ (a_mean_en - a_mean_de) ** 2 / K.exp(a_logvar_de), axis=-1)
	kl_z_loss = 0.5 * K.sum(- z_logvar_en - 1 \
		+ K.exp(z_logvar_en) + z_mean_en ** 2, axis=-1)
	return x_loss + y_loss + kl_a_loss + kl_z_loss 








