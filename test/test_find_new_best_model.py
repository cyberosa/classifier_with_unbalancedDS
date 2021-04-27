import sys
from pathlib import Path

cwd = Path.cwd()
main_dir = cwd.parent
sys.path.append(str(cwd))
sys.path.append(str(main_dir))

from find_new_best_model import *


class TestTrainModel():
	def test_train_eval_model(self):
		df = pd.read_pickle('training_df')
		# split the sets
		X_train, X_test, y_train, y_test = get_train_test_sets(df, 'down')
		best_p = find_best_model_parameters('rf_classifier', X_train, y_train)
		assert len(best_p.keys()) == 6
		p_0, r_0, p_1, r_1 = train_eval_model('rf_classifier', X_train, y_train, X_test, y_test, **best_p)
		# TODO you can change these values to evaluate if the best parameters are acceptable of not
		assert p_0 >= 0.70
		assert r_0 >= 0.80
		assert p_1 >= 0.82
		assert r_1 >= 0.00