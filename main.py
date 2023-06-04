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


app = Flask(__name__)

def preprocess_data(df):
    # Handle non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    for col in non_numeric_cols:
        df[col] = pd.factorize(df[col])[0] + 1  # Convert non-numeric values to numeric labels

    # Handle missing values
    df.fillna(0, inplace=True)  # Replace missing values with 0

    return df

def train_model(df, algorithm, target_variable, test_train_split):
    # Split data into train and test sets
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_split/100, random_state=42)

    # Train the selected algorithm
    if algorithm == 'decision_tree':
        model = DecisionTreeClassifier()
    elif algorithm == 'random_forest':
        model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X_test, y_test

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['csv_file']
        if file:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file)
            df = preprocess_data(df)
            
            # Get form data
            algorithm = request.form['algorithm']
            target_variable = request.form['target_variable']
            test_train_split = int(request.form['test_train_split'])

            model, X_test, y_test = train_model(df, algorithm, target_variable, test_train_split)
            joblib.dump(model, 'static/model.pkl')
            session['columns'] = list(df.columns)
            session['target_variable'] = target_variable

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)

            print(target_variable)

            # Evaluate the model
            accuracy = model.score(X_test, y_test)*100
            print(f"Accuracy: {accuracy}")
            print(target_variable)

            # Plot 1: Distribution of Target Variable
            sns.countplot(df[target_variable])
            plt.xlabel(target_variable)
            plt.ylabel('Count')
            plt.title('Distribution of Target Variable')
            plt.tight_layout()
            plt.savefig('static/countplot.png')  # Save the plot as a static image

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
    if request.method == 'POST':
        # Load the trained model
        model = joblib.load('static/model.pkl')

        # Get form data
        input_values = {}
        for column in columns:
            input_values[column] = [request.form[column]]

        # Create DataFrame for prediction
        input_df = pd.DataFrame.from_dict(input_values)
        print(input_df)

        # Preprocess input data
        input_df = preprocess_data(input_df)
        print(input_df)

        # Make prediction
        prediction = model.predict(input_df)
        output = "Good" if prediction[0] else "Average"

        return render_template('predict_input.html', columns=columns, prediction=output)

    return render_template('predict_input.html', columns=columns, target=target_variable)


if __name__ == '__main__':
    app.secret_key = 'somerandomkey'
    app.run(debug=True)
