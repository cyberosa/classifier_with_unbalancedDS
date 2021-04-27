import sys
from pathlib import Path

cwd = Path.cwd()
main_dir = cwd.parent
sys.path.append(str(cwd))
sys.path.append(str(main_dir))

from get_model_prediction import *

class TestGetModelPrediction():
	def test_non_converted_lead(self):
		params_dict = prepare_dict(11, 1, 10966, 5,
								   'Signup', 'Signup', 'Direct', 'Pricing',
								   'isp', 'developer')

		result = get_model_prediction('./models/rf_model.joblib',
							 params_dict, False)
		assert result == 0
		result2 = get_model_prediction('./models/rf_model.joblib',
							 params_dict, True)
		assert round(result2,3) == 0.099

	def test_converted_lead(self):
			params_dict = prepare_dict(12, 1, 17576, 167,
									   'Form Fill', 'WF | Pricing Contact', 'Direct',
									   'Solutions Use Cases',
									   'None', 'engineering')

			result = get_model_prediction('./models/rf_model.joblib',
										  params_dict, False)
			assert result == 1
			result2 = get_model_prediction('./models/rf_model.joblib',
										  params_dict, True)
			assert round(result2,3) == 0.884
