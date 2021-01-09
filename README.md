# Diabetes Early Detection
This the implementation of farly diabetes detection using machine learning. Rather than just creating the machine learning model, this application is the real implementation with three user interfaces: interactive text mode (CLI), web browser based (Web), and graphical/desktop based (GUI).

The algorithm used for the machine learning is SVM (Support Vector Machine). Before generating the machine learning model, the application used grid search cross validation for determining the most optimal parameter for the SVM algorithm. The trained model can also be saved into a file so the application can load it later and doesn't have to repeat the training process.

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

## Dataset source
The dataset for generating the machine learning model is taken from the following URL:

https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.

## Citation
* Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
* Islam, MM Faniqul, et al. 'Likelihood prediction of diabetes at early stage using data mining techniques.' Computer Vision and Machine Intelligence in Medical Image Analysis. Springer, Singapore, 2020. 113-125.
