from flask import Flask, jsonify, request
import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config["DEBUG"] = True

model = load_model('NCF.h5')
graph = tf.get_default_graph()
#####################
#
# Prepare data
#
#####################

df = pd.read_pickle('ncfdataset.p')

uo = np.asarray(df[['user', 'item', 'uvec', 'ivec']])
labels = np.asarray(df.interaction)
X_train, X_test, y_train, y_test = train_test_split(uo, labels, test_size=.2)
uix_test = X_test[:, 0]
oix_test = X_test[:, 1]
uvecs_test = X_test[:, 2]
ovecs_test = X_test[:, 3]
uvecs_test = np.matrix(uvecs_test.tolist())
ovecs_test = np.matrix(ovecs_test.tolist())

@app.route('/', methods=['GET','POST'])
def predict():
    with graph.as_default():
        pred = model.predict([uix_test, oix_test, uvecs_test, ovecs_test])    
   # pred = np.round(pred, decimals=0).astype(int)
    response = {}
    response['predictions'] = pred.tolist()
    return flask.jsonify(response)

if __name__ == '__main__':
    app.run()

