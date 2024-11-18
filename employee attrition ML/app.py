from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (Make sure model.pkl is in the same directory as app.py)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("employee_attrition.html")  # Make sure this template exists

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Get the input values from the form (satisfaction_level, average_monthly_hours)
        satisfaction_level = float(request.form['satisfaction_level'])
        average_monthly_hours = float(request.form['average_monthly_hours'])

        # Create a NumPy array for the prediction
        input_features = np.array([[satisfaction_level, average_monthly_hours]])

        # Make the prediction
        prediction = model.predict_proba(input_features)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)  # Probability of attrition

        # Interpret the result
        if float(output) > 0.5:
            return render_template(
                'employee_attrition.html',
                pred=f'The employee is likely to leave.\nProbability of attrition: {output}',
                advice="Consider taking retention measures to improve satisfaction and manage workload."
            )
        else:
            return render_template(
                'employee_attrition.html',
                pred=f'The employee is likely to stay.\nProbability of attrition: {output}',
                advice="Keep up the good work to maintain employee satisfaction!"
            )
    except Exception as e:
        return render_template('employee_attrition.html', pred=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

