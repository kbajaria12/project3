from __future__ import print_function
from flask import Flask, render_template, jsonify, redirect, request
import sagemaker
from sagemaker.mxnet import MXNetPredictor
from sagemaker.tensorflow import TensorFlowPredictor
import sys
import ast

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mxnet')
def mxnet():
    
    mynumber = request.args.getlist('image')

    predictor = MXNetPredictor('sagemaker-mxnet-2018-04-12-21-02-24-757')
    
    mynumberarray = ast.literal_eval(mynumber[0])
    
    response = predictor.predict(mynumberarray)
    
    labeled_predictions = list(zip(range(10), response[0]))

    labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
    answer= "Most likely answer: "+str(labeled_predictions[0])
    
    return(answer)

@app.route('/tensor')
def tensor():
    
    mynumber = request.args.getlist('image')

    predictor = TensorFlowPredictor('sagemaker-tensorflow-2018-04-15-02-09-49-209')
    
    mynumberarray = ast.literal_eval(mynumber[0])
    
    response = predictor.predict(mynumberarray)
    
    prediction = response['outputs']['classes']['int64Val'][0]
    print("prediction is {}".format(prediction))
    
    answer= ("Most likely answer: {}".format(prediction))
    
    return(answer)
   
if __name__ == "__main__":
    app.run(debug=False)
