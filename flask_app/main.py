"""
    File name: main.py
    Author: Kshitij Shah
    Date created: 5/31/2018
    Python Version: 3.6
"""


import os
from flask import Flask, request, Response, redirect, jsonify, render_template
from models import FFNPredictor

app = Flask(__name__)

"""
FFN Predictor uses a feed forward neural network on TF-IDF features. 
The pretrained model is provided from ffn.h5 file and the TF-IDF vectorizer from vectorizer.pkl
This class provides the vital 'predict' method which returns both prediction and confidence given a string of words
"""
predictor = FFNPredictor(os.path.join(app.root_path, 'saved_models/ffn.h5'),
                         os.path.join(app.root_path, 'saved_models/vectorizer.pkl'))


@app.route('/_process_input')
def process_input():
    """
    This method processes the input words provided by web interface using jquery
    :return: a json object with prediction and confidence
    """
    words = request.args.get('words', 0, type=str)
    prediction, confidence = predictor.predict(str(words))
    data = {'prediction': prediction,
            'confidence': f"{confidence:.4f}"}
    resp = jsonify(data)
    resp.status_code = 200
    return resp


@app.route('/index')
def render_index():
    """
    Render index page for the web interface
    :return: rendered html document
    """
    return render_template('single_document.html')


@app.route('/', methods=['GET'])
def handle_get():
    """
    Handles get request and provides response in accepted mimetype
    :return: Response object with data and status
    """
    accepted_types = [x.strip().split(';')[0] for x in request.headers['Accept'].split(',')]
    print(accepted_types)
    if 'words' in request.args:
        if not request.args['words']:
            return Response('Empty document can not be processed', status=400, mimetype='text/plain')
        for accept in accepted_types:
            print(accept)
            if accept in ['text/plain', 'text/html', 'text/csv', '*/*']:
                prediction, confidence = predictor.predict(request.args['words'])
                return Response(f"{prediction}, {confidence:.4f}",
                                status=200,
                                mimetype=accept)
            if accept in ['application/json']:
                prediction, confidence = predictor.predict(request.args['words'])
                data = {'prediction': prediction,
                        'confidence': float(confidence)}
                resp = jsonify(data)
                resp.status_code = 200
                return resp
            if accept in ['application/xml']:
                prediction, confidence = predictor.predict(request.args['words'])
                return Response(f'<?xml version="1.0" ?>'
                                f'<response>'
                                f'<prediction>{prediction}</prediction>'
                                f'<confidence>{confidence:.4f}</confidence>'
                                f'</response>', status=200, mimetype=accept)
        return Response("Only the following types are supported. text/plain', 'text/html', 'text/csv, application/json, application/xml",
                        status=406,
                        mimetype='text/plain')
    else:
        return redirect('/index')


if __name__ == '__main__':
    app.run()
