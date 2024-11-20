from flask import Flask,render_template, request
import pickle
import numpy as np

model_path = r'C:\ml_project_files\model1.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
print("Model loaded successfully.")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    cgpa = float(request.form.get('cgpa'))
    iq = int(request.form.get('iq'))
    profile_score = int(request.form.get('profile_score'))

    # prediction
    result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))

    if result[0] == 1:
        result = 'placed'
    else:
        result = 'not placed'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    print(__name__)
    app.run(host='0.0.0.0', port=8082)
