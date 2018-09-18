"""Train a recommendation engine"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split


#####################
#
# Prepare data
#
#####################

df = pd.read_pickle('ncfdataset.p')

uo = np.asarray(df[['user', 'item', 'uvec', 'ivec']])
labels = np.asarray(df.interaction)
X_train, X_test, y_train, y_test = train_test_split(uo, labels, test_size=.2)

uix_train = X_train[:, 0]
uix_test = X_test[:, 0]
oix_train = X_train[:, 1]
oix_test = X_test[:, 1]
uvecs_train = X_train[:, 2]
uvecs_test = X_test[:, 2]
ovecs_train = X_train[:, 3]
ovecs_test = X_test[:, 3]
uvecs_train = np.matrix(uvecs_train.tolist())
ovecs_train = np.matrix(ovecs_train.tolist())
uvecs_test = np.matrix(uvecs_test.tolist())
ovecs_test = np.matrix(ovecs_test.tolist())

#####################
#
# Model parameters
#
#####################

num_users = uvecs_train.shape[0]
num_items = ovecs_train.shape[0]
batch_size = 256
epochs = 10
learning_rate = 0.001
layers = [200, 10]
reg_layers = [0, 0]
latent_dim = 8
regs = [0, 0]
uvec_lenght = df.uvec.values[0].shape[0]
ovec_lenght = df.ivec.values[0].shape[0]

#####################
#
# Model definition
#
#####################

gmf_user_input = Input(shape=(1,), dtype='int32', name='gmf_user_input')
gmf_open_input = Input(shape=(1,), dtype='int32', name='gmf_open_input')

mlp_user_input = Input(shape=(uvec_lenght,), dtype='float32',
                       name='mlp_user_input', batch_shape=(None, uvec_lenght))
mlp_open_input = Input(shape=(ovec_lenght,), dtype='float32',
                       name='mlp_open_input', batch_shape=(None, ovec_lenght))

MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                              embeddings_initializer='he_uniform', embeddings_regularizer=l2(regs[0]), input_length=1)
MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                              embeddings_initializer='he_uniform', embeddings_regularizer=l2(regs[1]), input_length=1)

user_latent = Flatten()(MF_Embedding_User(gmf_user_input))
item_latent = Flatten()(MF_Embedding_Item(gmf_open_input))

gmf_predict_vector = keras.layers.Multiply()([user_latent, item_latent])

mlp_full_input = keras.layers.Concatenate(
    name='mlp_full_input')([mlp_user_input, mlp_open_input])

dense1 = Dense(layers[0], kernel_regularizer=l2(reg_layers[0]), activation='relu',
               name='dense1')(mlp_full_input)
dense2 = Dense(layers[1], kernel_regularizer=l2(reg_layers[1]), activation='relu',
               name='dense2')(dense1)

predict_vector = keras.layers.Concatenate(
    name='predict_vector')([gmf_predict_vector, dense2])

ncf_prediction = Dense(1, activation='sigmoid',
                       kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

model_ncf = Model(inputs=[gmf_user_input, gmf_open_input,
                          mlp_user_input, mlp_open_input], outputs=ncf_prediction)

model_ncf.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'mean_absolute_error'])

#####################
#
# Model training
#
#####################

hist = model_ncf.fit([uix_train, oix_train, uvecs_train, ovecs_train],
                     y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     shuffle=True,
                     class_weight={0: .2, 1: 1})

#####################
#
# Evaluation
#
#####################

y_pred = hist.model.predict([uix_test, oix_test, uvecs_test, ovecs_test])
y_pred = np.round(y_pred, decimals=0).astype(int)

print(classification_report(y_test, y_pred.flatten()))

print(confusion_matrix(y_test, y_pred.flatten()))

#####################
#
# Store model
#
#####################

hist.model.save('NCF.h5')
