# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
from data_processing import *
from get_model_prediction import *

# creating a Flask app
app = Flask(__name__)

def get_features_dict(args_dict,
					  numerical_cols = ['place_within_tenant','city','region_code']):
	'''
	Given the dictionary with the arguments received with the request,
	prepare the dictionary with the proper data types for
	the prediction. Cast numerical and boolean values.
	:param args_dict:
	:param numerical_cols:
	:return:
	'''
	if not set(numerical_cols).issubset(set(args_dict.keys())):
		print("Error missing arguments in the dictionary")
		return None
	try:
		# by default all args are parsed as strings
		# so we need to cast the numerical values into ints
		# besides we need to rename some fields
		args_dict['month'] = int(args_dict['month'])
		args_dict['place_within_tenant'] = int(args_dict['place_within_tenant'])
		args_dict['city'] = int(args_dict['city'])
		args_dict['region_code'] = int(args_dict['region_code'])
		args_dict['scoring'] = True if args_dict['scoring']=='True' else False
		return args_dict
	except Exception as e:
		print("Error when casting numerical arguments")
		return None

# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods = ['GET', 'POST'])
def home():
	if(request.method == 'GET'):
		intro = "Welcome to the leads conversion score system. Use the following format to send your request"
		# order of the parameters
		#	params_d = dict()
			# params_d['month'] = month
			# params_d['place_within_tenant'] = place_within_tenant
			# params_d['city'] = city
			# params_d['region_code'] = region_code
			# params_d['lead_source'] = lead_source
			# params_d['campaign_name'] = campaign_name
			# params_d['group_source'] = group_source
			# params_d['group_landing'] = group_landing
			# params_d['type'] = type
			# params_d['role'] = role
		reqs = 'curl http://127.0.0.1:5000/pred?month=6&place_within_tenant=1' \
			   '&city=10050&region_code=333&lead_source=Signup&campaign_name=Signup' \
			   '&group_source=Google&group_landing=Home' \
			   '&type=None&role=non-developer&scoring=True'
		return jsonify({'intro': intro, 'reqs':reqs})


# A simple function to get the predictions of a model
# the needed variables are sent in the URL when we use GET
@app.route('/pred', methods = ['GET'])
def predict_conversion():
	try:
		pred_params = request.args.to_dict()
		if len(pred_params.keys()) < 11:
			msg = "Missing parameters in the request"
			return jsonify({'error_message': msg}), 400

		pred_params = get_features_dict(pred_params)

		if pred_params is None:
			msg = "Wrong or missing parameters in the request"
			return jsonify({'error_message': msg}), 400

		result = get_model_prediction('models/rf_model.joblib',
								  pred_params, pred_params['scoring'])

		if result is None:
			msg = "There was an error when computing the prediction. Please check your parameters"
			return jsonify({'message': msg}), 400

		conversion = False
		if result > 0.5:
			conversion = True
		return jsonify({'potential_conversion': conversion,
						'lead_score': result})
	except Exception as e:
		msg = "There was an error when processing the request"
		return jsonify({'message': msg, 'error':e}), 400


# driver function
if __name__ == '__main__':

	app.run(debug = True)
