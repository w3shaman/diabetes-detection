# Diabetes Early Detection
This the implementation of early diabetes detection using machine learning. Rather than just creating the machine learning model, this application is the real implementation with three user interfaces: interactive text mode (CLI), web browser based (Web),  graphical/desktop based (GUI), and also API mode (RESTAPI). From those four interfaces, user must input the age, gender, then choose some known diabetes symptoms. After submitting the symptoms, the application will predict the result if it's POSITIVE or NEGATIVE.

The algorithm used for the machine learning is SVM (Support Vector Machine). Before generating the machine learning model, the application used grid search cross validation for determining the most optimal parameter for the SVM algorithm. The trained model can also be saved into a file so the application can load it later and doesn't have to repeat the training process.

## Disclaimer
This application is for learning purpose only. The prediction result may not meet your expectation.

## Usage
The complete arguments for running the application can be viewed using the following command.

``
python diabetes_early_detection.py --help
``

To start the application in interactive text mode.

``
python diabetes_early_detection.py --mode cli
``

To start the application in graphical or desktop based.

``
python diabetes_early_detection.py --mode gui
``

To start the application in web browser based. Then open **http://localhost:5000** on your web browser.

``
python diabetes_early_detection.py --mode web
``

To start the application in REST API mode. Then open **http://localhost:5000** on your web browser to see the simple documentation about using the API.

``
python diabetes_early_detection.py --mode restapi
``

## Save and Load Training Model
We can also save then load the training model later so the application doesn't have to do the training process on every start.

For saving the generated training model, please add the following parameter when running the application.

``
--generate-pickle models/diabetes_early_detection.pkl
``

For loading the generated training model, we can add the following parameter instead of the above.

``
--load-pickle models/diabetes_early_detection.pkl
``

## Dataset source
The dataset for generating the machine learning model is taken from the following URL:

https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.

## Citation
* Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
* Islam, MM Faniqul, et al. 'Likelihood prediction of diabetes at early stage using data mining techniques.' Computer Vision and Machine Intelligence in Medical Image Analysis. Springer, Singapore, 2020. 113-125.
