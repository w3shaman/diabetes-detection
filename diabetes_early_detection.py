'''
Diabetes early detection system using Support Vector Machine
and Grid Search for fine tuning the parameters.

Dataset is taken from:
https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
'''

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import pandas as pd
import argparse
import pickle


'''
Parsing CLI arguments.
'''
parser = argparse.ArgumentParser()
parser.add_argument("--generate-pickle", help="Set the location for storing pickle (the file of machine learning model). If omitted, the model will not be saved.", dest="generate_pickle", default=None)
parser.add_argument("--dataset", help="The dataset location. Default to: dataset/diabetes_data_upload.csv", dest="dataset_location", default="dataset/diabetes_data_upload.csv")
parser.add_argument("--load-pickle", help="Load pickle (saved machine learning model) instead of retraining. This should be location of the file. If omitted the application will repeat the training process.", dest="load_pickle", default=None)
parser.add_argument("--mode", help="The application mode (Web, GUI, or CLI).", dest="app_mode", default='cli')
parser.add_argument("--verbose", help="The verbose mode. Show the machine learning statistics and application message when running.", dest="verbose", default='no')
args = parser.parse_args()


'''
Load the dataset.
'''
df = pd.read_csv(args.dataset_location)

cols = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity"
]

for col in cols:
    df[col] = df[col].map({"Yes":1, "No":0})

df["Gender"] = df["Gender"].map({"Male":1, "Female":0})
df["class"] = df["class"].map({"Positive":1, "Negative":0})

X = df.iloc[:, 0:16]
Y = df.iloc[:, 16]

'''
Split training data (70%) and test data (30%).
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


'''
Retrain or use generated model.
'''
if args.load_pickle == None:
    '''
    Parameter alternatives.
    '''
    param_grid = [
            {
                'kernel': ['linear'],
                'C': [1.0, 10.0, 100.0]
            },
            {
                'kernel': ['rbf'],
                'C': [1.0, 10.0, 100.0],
                'gamma': [0.1, 0.2, 0.5]
            }
        ]

    verbose = 0
    if args.verbose == 'y':
        verbose = 1

    gs_svc = GridSearchCV(SVC(random_state=None), param_grid, scoring='accuracy', cv=5, verbose=verbose)
    gs_svc.fit(X_train, Y_train)

    '''
    Best parameter for training model.
    '''
    if args.verbose == 'y':
        print('Best parameter set: %s ' % gs_svc.best_params_)
        print('CV Accuracy: %.3f' % gs_svc.best_score_)

    clf = gs_svc.best_estimator_
else:
    clf = pickle.load(open(args.load_pickle, 'rb'))

    if args.verbose == 'y':
        print('Training model loaded from: ' + args.load_pickle);


'''
Display the training model performance.
'''
if args.verbose == 'y':
    print("Training accuracy: %.3f" % clf.score(X_train, Y_train))
    print("Testing accuracy: %.3f" % clf.score(X_test, Y_test))


'''
Save the training model.
'''
if args.generate_pickle != None and args.load_pickle == None:
    pickle.dump(clf, open(args.generate_pickle, 'wb'), protocol = 4)

    if args.verbose == 'y':
        print('Training model saved to: ' + args.generate_pickle);


'''
Load CLI or Web application.
'''
if args.app_mode.lower() == 'cli':
    '''
    Predict based on user's input.
    '''
    data = []

    print('')
    print('Diabetes early detection')
    print('------------------------')

    i = input('Age: ')
    data.append(int(i))

    i = input('Gender (m/f): ')
    if i.lower() == 'm':
        data.append(1)
    else:
        data.append(0)

    for col in cols:
        i = input(col + ' (y/n): ')
        if i.lower() == 'y':
            data.append(1)
        else:
            data.append(0)

    label = clf.predict([data])

    print('')

    if label[0] == 0:
        print('Prediction: NEGATIVE')
    else:
        print('Prediction: POSITIVE')

    print('')
elif (args.app_mode.lower() == 'web'):
    '''
    Web application using Flask.
    '''
    from flask import Flask, request, render_template

    app = Flask(__name__)

    @app.route('/', methods=['POST', 'GET'])
    def diabetes_detection():
        prediction = ''
        error = ''
        post_data = {}

        try:
            if request.method == 'POST':
                post_data['age'] = request.form['age']
                post_data['gender'] = request.form['gender']

                data = []

                data.append(int(request.form['age']))

                if request.form['gender'].lower() == 'm':
                    data.append(1)
                else:
                    data.append(0)

                for i, col in enumerate(cols):
                    post_data['s' + str(i)] = request.form['s' + str(i)]

                    if request.form['s' + str(i)].lower() == 'y':
                        data.append(1)
                    else:
                        data.append(0)

                label = clf.predict([data])
                if label[0] == 0:
                    prediction = 'Prediction: NEGATIVE'
                else:
                    prediction = 'Prediction: POSITIVE'
        except Exception as e:
            error = str(e)

        return render_template('index.html', symptoms=cols, prediction=prediction, post_data=post_data, error=error)

    '''
    Use gevent for as web server with the capabilities of
    handling asynchronous requests
    '''
    from gevent import monkey
    monkey.patch_all()

    debug = False
    if args.verbose == 'y':
        debug = True

    '''
    Here we start the WSGI server.
    '''
    from gevent.pywsgi import WSGIServer

    print('Web server listening on port 5000.')
    app.debug = debug
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
elif (args.app_mode.lower() == 'restapi'):
    '''
    REST API using Flask.
    '''
    from flask import Flask

    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def help():
        from flask import render_template

        return render_template('restapi-help.html')

    @app.route('/', methods=['POST'])
    def diabetes_detection():
        from flask import request, jsonify

        try:
            data = request.get_json()

            label = clf.predict([data])
            if label[0] == 0:
                prediction = 'NEGATIVE'
            else:
                prediction = 'POSITIVE'

            return jsonify({'result': prediction})
        except Exception as e:
            return jsonify({'result': 'ERROR: ' + str(e)}), 500

    '''
    Use gevent for as web server with the capabilities of
    handling asynchronous requests
    '''
    from gevent import monkey
    monkey.patch_all()

    debug = False
    if args.verbose == 'y':
        debug = True

    '''
    Here we start the WSGI server.
    '''
    from gevent.pywsgi import WSGIServer

    print('Web server listening on port 5000.')
    app.debug = debug
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
else:
    import tkinter as tk

    app = tk.Tk()
    app.title("Diabetes Early Detection")

    lAge = tk.Label(app, text = "Age")
    lAge.grid(row = 1, column = 1)
    eAge = tk.Entry(app, bd = 5)
    eAge.grid(row = 1, column = 2)

    gender = tk.StringVar()
    lGender = tk.Label(app, text = "Gender")
    lGender.grid(row = 2, column = 1)
    rGenderM = tk.Radiobutton(app, text="Male", value="m", var=gender)
    rGenderM.grid(row = 2, column = 2)
    rGenderF = tk.Radiobutton(app, text="Female", value="f", var=gender)
    rGenderF.grid(row = 2, column = 3)

    symptoms = []
    lSymptom = []
    rSymptomY = []
    rSymptomN = []
    for i, col in enumerate(cols):
        symptoms.append(tk.StringVar())
        lSymptom.append(tk.Label(app, text = col))
        lSymptom[i].grid(row = i + 3, column = 1)
        rSymptomY.append(tk.Radiobutton(app, text="Yes", value="y", var=symptoms[i]))
        rSymptomY[i].grid(row = i + 3, column = 2)
        rSymptomN.append(tk.Radiobutton(app, text="No", value="n", var=symptoms[i]))
        rSymptomN[i].grid(row = i + 3, column = 3)

    def predict_result():
        from tkinter import messagebox

        try:
            data = []

            data.append(int(eAge.get()))

            if gender.get().lower() == 'm':
                data.append(1)
            else:
                data.append(0)

            for i, col in enumerate(cols):
                if symptoms[i].get().lower() == 'y':
                    data.append(1)
                else:
                    data.append(0)
            label = clf.predict([data])
            if label[0] == 0:
                prediction = 'NEGATIVE'
            else:
                prediction = 'POSITIVE'

            messagebox.showinfo( "Prediction", prediction)
        except Exception as e:
            error = str(e)
            messagebox.showinfo( "Error", error)

    bPredict = tk.Button(app, text="PREDICT RESULT", command=predict_result)
    bPredict.grid(row = i + 4, column = 1)

    app.mainloop()
