from flask import Flask, render_template, request
import pickle
import numpy as np
import os

model_path = r'C:\ml_project_files\model1.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result=None)  # Ensuring a clean start

@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_score = int(request.form.get('profile_score'))

        if cgpa < 0 or iq < 0 or profile_score < 0:
            return render_template('index.html', result="⚠️ Please enter positive values.")


        result = model.predict(np.array([[cgpa, iq, profile_score]]))  # Using double brackets for proper reshaping
        result_text = "🎉 Congratulations! You are likely to be Placed. ✅" if result[0] == 1 else "❌ Unfortunately, Placement is Unlikely. Keep Improving!"

        return render_template('index.html', result=result_text)

    except ValueError:
        return render_template('index.html', result="⚠️ Invalid Input! Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)


