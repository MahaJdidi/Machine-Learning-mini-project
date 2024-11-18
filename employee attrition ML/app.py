from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("employee_attrition.html")  

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        
        satisfaction_level = float(request.form['satisfaction_level'])
        average_monthly_hours = float(request.form['average_monthly_hours'])

        
        input_features = np.array([[satisfaction_level, average_monthly_hours]])

        
        prediction = model.predict_proba(input_features)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)  # Probability of attrition

        
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

