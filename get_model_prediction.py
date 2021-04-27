import joblib
from data_processing import *


def prepare_dict(month, place_within_tenant,
				 	city, region_code, lead_source,
					campaign_name, group_source,
					group_landing, type, role):
	'''
	Method to wrap all model predictors in a dictionary.
	A potential way to optimize it would be spliting the numerical and categorical variables
	in separate dictionaries
	:param month:
	:param place_within_tenant:
	:param city:
	:param region_code:
	:param lead_source:
	:param campaign_name:
	:param group_source:
	:param group_landing:
	:param type:
	:param role:
	:return:
	'''
	params_d = dict()
	params_d['month'] = month
	params_d['place_within_tenant'] = place_within_tenant
	params_d['city'] = city
	params_d['region_code'] = region_code
	params_d['lead_source'] = lead_source
	params_d['campaign_name'] = campaign_name
	params_d['group_source'] = group_source
	params_d['group_landing'] = group_landing
	params_d['type'] = type
	params_d['role'] = role
	return params_d

def get_model_prediction(model_filename,params_dict,prob_score):
	'''
	Given the parameters for the model as a dictionary, get a prediction from the model.
	For the moment it work only with one request at a time, i.e., only one value per key.
	:param model_filename: path to the file with the model
	:param params_dict: dictionary with the needed predictors for the model
	:param prob_score: True if the prediction should be a probability instead of a label
	:return:
	'''
	try:
		# check the length of the dictionary and missing keys
		if len(params_dict.keys()) < 10:
			print("length of params_dict{}".format(len(params_dict.keys())))
			print("Error missing arguments for the model")
			return None

		needed_values = ['month','place_within_tenant','city',
							'region_code', 'lead_source',
							'campaign_name','group_source',
							 'group_landing','type','role']
		#print(params_dict.keys())
		if not set(needed_values).issubset(set(params_dict.keys())):
			print("Error missing arguments for the model")
			return None

		# load
		print("Model filename {}".format(model_filename))
		model = joblib.load(model_filename)

		#print("Received parameters {}".format(params_dict))

		print("Scaling numerical values")
		# perform transformations on the numerical cols
		num_vars = scale_num_features(params_dict['place_within_tenant'],
							   params_dict['city'],
							   params_dict['region_code'],
							   tenant_range=[1, 261],
							   city_range=[0, 27332],
							   region_range=[0, 391])

		#place_within_tenant_mm, city_mm, region_code_mm = num_vars
		print("scaled num variables {}".format(num_vars))

		print("Transforming categorical values")
		# performm transformations on the categorical cols
		cat_vars = process_cat_features(
			params_dict['lead_source'],
			params_dict['campaign_name'],
			params_dict['group_source'],
			params_dict['group_landing'],
			params_dict['type'],
			params_dict['role'])

		print("transformed cat variables {}".format(cat_vars))
		#lead_source_cat, campaign_name_cat, group_source_cat, group_landing_cat, type_cat, role_cat = cat_vars
		# X_test structure
		X_test = pd.DataFrame(columns=['month',
							 'place_within_tenant_mm',
							 'city_mm',
							 'region_code_mm',
							 'lead_source_cat',
							 'campaign_name_cat',
							 'group_source_cat',
							 'group_landing_cat',
							 'type_cat',
							 'role_cat'])

		# append data to X_test
		x_params = [params_dict['month']]
		x_params.extend(num_vars)
		x_params.extend(cat_vars)
		X_test.loc[0,:] = x_params
		#print(X_test)
		if len(X_test) == 0:
			print("Error with the input dataframe")
			return None

		# generate the prediction
		print("Computing prediction")
		if prob_score:
			result = model.predict_proba(X_test)
			y_pred = float(result[0][1])
		else:
			result = model.predict(X_test)
			y_pred = int(result[0])
		return y_pred
	except Exception as e:
		print(e)
		print("Error when trying to generate the prediction")
		return None

if __name__ == '__main__':
	try:
		# model_filename plus 10 predictors
		if len(sys.argv) < 11:
			print("error missing parameters")
			sys.exit()

		model_filename = str(sys.argv[1])
		month = int(sys.argv[2])
		place_within_tenant = int(sys.argv[3])
		city = int(sys.argv[4])
		region_code = int(sys.argv[5])
		lead_source = str(sys.argv[6])
		campaign_name = str(sys.argv[7])
		group_source = str(sys.argv[8])
		group_landing = str(sys.argv[9])
		type = str(sys.argv[10])
		role = str(sys.argv[11])

		params_dict = prepare_dict(month, place_within_tenant,
								   city, region_code, lead_source,
								   campaign_name, group_source,
								   group_landing, type, role)

		result = get_model_prediction(model_filename,
							 params_dict, True)
		print(result)
	except Exception as e:
		print(e)
