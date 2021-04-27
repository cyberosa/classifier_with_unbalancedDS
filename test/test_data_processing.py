import sys
from pathlib import Path

cwd = Path.cwd()
main_dir = cwd.parent
sys.path.append(str(cwd))
sys.path.append(str(main_dir))

from data_processing import *


class TestScaleNumFeatures:
    def test_scale_num_features(self):
        # test within range
        t,c,r = scale_num_features(1, 927, 258)
        assert t == 0.0
        assert round(c,3) == 0.034
        assert round(r,3) == 0.66

        # tests outside range
        t,c,r = scale_num_features(262, 7335, 3)
        assert t == 1.0
        assert round(c,3) == 0.268
        assert round(r,3) == 0.008

        t,c,r = scale_num_features(32, -1, -1)
        assert round(t,3) == 0.119
        assert round(c,3) == 0.0
        assert round(r,3) == 0.0

        t,c,r = scale_num_features(7, 3600, 392)
        assert round(t, 3) == 0.023
        assert round(c,3) == 0.132
        assert round(r,3) == 1.0


class TestProcessCatFeatures:
    def test_process_cat_features(self):
        v1, v2, v3, v4, v5, v6 = process_cat_features('missing', 'Signup', 'missing', 'luke', 'isp', 'r2d2')
        assert v1 == 29
        assert v2 == 165
        assert v3 == 62
        assert v4 == 80
        assert v5 == 1
        assert v6 == 87
        v1, v2, v3, v4, v5, v6 = process_cat_features('Gated Content', 'missing','dark vader',
                                                      'Dashboard Auth', 'missing', 'minor')
        assert v1 == 8
        assert v2 == 318
        assert v3 == 62
        assert v4 == 11
        assert v5 == 2
        assert v6 == 119
        v1, v2, v3, v4, v5, v6 = process_cat_features('the child', 'mando', 'Direct',
                                                      'missing', 'far', 'engineering')
        assert v1 == 29
        assert v2 == 318
        assert v3 == 10
        assert v4 == 80
        assert v5 == 2
        assert v6 == 60