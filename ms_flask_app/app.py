from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
import traceback
import ast
import sklearn
import xgboost
pickledModel = pickle.load(open('../app/public/latePaymentsModel.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/process',methods=["POST"])
def process():
	if request.method == 'POST':
		payLatePrediction = ast.literal_eval(request.form['rawtext'])
		try:
			payLatePredictionDf = pd.DataFrame.from_dict(payLatePrediction)
			result = pickledModel.predict_proba(payLatePredictionDf)
		except:
			traceback.print_exc()
			result = "Oops! Something went wrong"
	
	return render_template("index.html",result='Approved!' if result[0][0] >= 0.5 else 'Rejected!')


if __name__ == '__main__':
	app.run(debug=True)
