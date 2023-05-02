from flask import Flask, request, jsonify
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

app = Flask(__name__)


with open('MlModel/EProcurementModel.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    experience = int(data.get('experience'))
    successfulTenders = int(data.get('successfulTenders'))
    subCategory = int(data.get('subCategory'))
    budget = int(data.get('budget'))
    quotigPrice = int(data.get('quotigPrice'))

    prediction = model.predict([[experience,successfulTenders,subCategory,budget,quotigPrice]])
    prediction = round(prediction[0],2)

    return jsonify({'prediction': prediction})


if __name__=='__main__':
    app.run(debug=True)