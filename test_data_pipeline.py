import unittest
import pandas as pd
import numpy as np
import toolbox


class TestMain(unittest.TestCase):

    def test_get_init_pos1(self):
        dict_ = {'frame': [0, 1], 'x': [1, 2], 'y': [2, 3]}
        df = pd.DataFrame(dict_)
        init_pos = toolbox.get_pos_at_frame(df, 0)
        self.assertIsInstance(init_pos, np.ndarray)

    def test_get_init_pos2(self):
        dict_ = {'frame': [0, 1], 'x': [1, 2], 'y': [2, 3]}
        df = pd.DataFrame(dict_)
        init_pos = toolbox.get_pos_at_frame(df, 0)
        self.assertEqual(init_pos.shape[0], 2)

    def test_get_init_pos3(self):
        dict_ = {'frame': [0, 1], 'x': [1, 2], 'y': [2, 3]}
        df = pd.DataFrame(dict_)
        init_pos = toolbox.get_pos_at_frame(df, 0)
        self.assertTrue(init_pos.shape[1] > 1)


if __name__ == '__main__':
    unittest.main()
