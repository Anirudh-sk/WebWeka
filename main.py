import pandas as pd
from flask import Flask, render_template, request, session
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os
import time
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression




app = Flask(__name__)
app.secret_key = "anirudh"
app.config.update(SECRET_KEY=os.urandom(24))


def preprocess_data(df):
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in non_numeric_cols:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col]) + 1 

    df.fillna(0, inplace=True)  

    return df

def preprocess_user_input(df):
    print(df.dtypes)

    numeric_cols = list(set(df.columns) - set(df.select_dtypes(include=['object']).columns))
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    print(numeric_cols)

    non_numeric_cols = df.select_dtypes(include=['object']).columns
    print(non_numeric_cols)
    le = LabelEncoder()
    for col in non_numeric_cols:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col]) + 1  

    return df

def train_model(df, algorithm, target_variable, test_train_split):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_split/100, random_state=42)

    if algorithm == 'decision_tree':
        model = DecisionTreeClassifier()
    elif algorithm == 'random_forest':
        model = RandomForestClassifier()
    elif algorithm == 'naive_bayes':
        model = GaussianNB()
    elif algorithm == 'logistic_regression':
        model = LogisticRegression()
    elif algorithm == 'linear_regression':
        model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csv_file']
        if file:
            dataset_name = file.filename.rsplit('.', 1)[0]
            df = pd.read_csv(file)
            df = preprocess_data(df)
            print(df.head())

            algorithm = request.form['algorithm']
            target_variable = request.form['target_variable'].strip()  
            test_train_split = int(request.form['test_train_split'])


            model, X_test, y_test = train_model(df, algorithm, target_variable, test_train_split)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_filename = f"{dataset_name}_{algorithm}_{timestamp}.pkl"
            joblib.dump(model, os.path.join('static', model_filename))

            # joblib.dump(model, 'static/model.pkl')
            session['columns'] = list(df.columns)
            session['target_variable'] = target_variable

            y_pred = model.predict(X_test)

            if algorithm == 'linear_regression':
                cm=model.score(X_test, y_test)*100
            else:
                cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)

            print(target_variable)

            accuracy = model.score(X_test, y_test)*100
            print(f"Accuracy: {accuracy}")
            print(target_variable)

            # Plot 1: Distribution of Target Variable
            sns.countplot(df[target_variable])
            plt.xlabel(target_variable)
            plt.ylabel('Count')
            plt.title('Distribution of Target Variable')
            plt.tight_layout()
            plt.savefig('static/countplot.png')  

            # Plot 2: Kernel Density Estimate
            plt.figure()
            sns.kdeplot(df[target_variable])
            plt.xlabel(target_variable)
            plt.ylabel('Density')
            plt.title('Kernel Density Estimate')
            plt.tight_layout()
            plt.savefig('static/kdeplot.png')

            # Plot 3: Pairplot
            plt.figure()
            sns.pairplot(df)
            plt.tight_layout()
            plt.savefig('static/pairplot.png')

            return render_template('index.html', accuracy=accuracy, graph='countplot.png', cm=cm,graph1='countplot.png', graph2='kdeplot.png', graph3='pairplot.png')

    return render_template('index.html')

@app.route('/predictinput', methods=['GET', 'POST'])
def predict_input():
    columns = session.get('columns', [])
    target_variable = session.get('target_variable', [])
    columnsdf = pd.DataFrame(columns)
    print(columnsdf)
    numeric_cols = set(columns) - set(columnsdf.select_dtypes(include=['object']).columns)
    print(numeric_cols)
    model_files = [file for file in os.listdir('static') if file.endswith('.pkl')]

    if request.method == 'POST':

        selected_model = request.form.get('selected_model')
        model_path = os.path.join('static', selected_model)
        model = joblib.load(model_path)

        # model = joblib.load('static/model.pkl')

        input_values = {}
        for column in columns:
            if column != target_variable:
                value = request.form.get(column)
                if column in numeric_cols:
                    try:
                        value = pd.to_numeric(value)
                    except ValueError:
                        pass
                input_values[column] = [value]

        input_df = pd.DataFrame.from_dict(input_values)
        print(input_df)

        input_df = preprocess_data(input_df)
        print(input_df)

        prediction = model.predict(input_df)
        print(prediction)
        output = prediction[0] 
        print(output)

        return render_template('predict_input.html', columns=columns, prediction=output)

    return render_template('predict_input.html', columns=columns, model_files=model_files , target=target_variable)



if __name__ == '__main__':
    app.secret_key='anirudh'
    
    app.run()
