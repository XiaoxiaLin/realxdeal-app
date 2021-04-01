from flask import Flask, jsonify, request, make_response, redirect, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import sklearn
import json
import joblib
import os
import logging

# param
UPLOAD_FOLDER = r"uploads"
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logging.basicConfig(
    filename='flask_log.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)


# HOME page that allow users to use single or batch prediction
@app.route('/')
def upload_page():
    return render_template('home.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/batch_prediction', methods=['GET', 'POST'])
def batch_prediction():
    """
    Batch Prediction with a csv file
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # load model and get predictions
        joblib_file = "model.joblib"  
        joblib_model = joblib.load(joblib_file)

        try:
            predictions= joblib_model.predict(df[['x']])
            print(predictions)
        except:
            return "Error occured!Please upload a csv file with the correct format!"
        print("done")

        response={'y':[]}
        response['y']=list(predictions)

        return make_response(jsonify(response),200)

    else: 
        return redirect(request.url)
    

@app.route("/single_prediction", methods=['GET', 'POST'])
def single_prediction():
    """
    Single Prediction. User input a number and will get the prediction.
    """
    x = request.form.get("x")
    print("Input number", x)   

    # load model and get predictions
    joblib_file = "model.joblib"  
    joblib_model = joblib.load(joblib_file)

    try:
        predictions= joblib_model.predict([[float(x)]])
        print("Prediction: ", predictions[0])
    except:
        return "Please enter a valid number for prediction!"
    print("done")

    response={'y':[]}
    response['y']=list(predictions)

    return make_response(jsonify(response),200)



@app.route("/predict", methods=['GET', 'POST'])
def predict():
    """
    Single prediction. 
    Allow user to easily get prediction by changing the parameter in the url. e.g.
    http://0.0.0.0:5000/predict?x=6
    """
    x = request.args.get('x', default=None, type=float)
    print("Input number", x) 

    # load model and get predictions
    joblib_file = "model.joblib"  
    joblib_model = joblib.load(joblib_file)

    try:
        predictions= joblib_model.predict([[x]])
        print("Prediction: ", predictions)
    except:
        return "Please enter a valid number for prediction!"
    print("done")

    response={'y':[]}
    response['y']=list(predictions)

    return make_response(jsonify(response),200)



@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    """
    Batch prediction with json data.
    """
    data = request.get_json()
    print("Input data", data) 
    df=pd.DataFrame(data['data'])

    # load model and get predictions
    joblib_file = "model.joblib"  
    joblib_model = joblib.load(joblib_file)

    try:
        predictions= joblib_model.predict(df[['x']])
        print("Prediction: ", predictions)
    except:
        return "Please enter a valid number for prediction!"
    print("done")

    response={'y':[]}
    response['y']=list(predictions)

    return make_response(jsonify(response),200)


if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)