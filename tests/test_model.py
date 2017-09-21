import unittest
from octane.model import Model


# dummy test
class TestModel(unittest.TestCase):
    def test_model(self):
        model = Model([2, 5, 1], 0.01, 500)
        with model:
            model.train((
                [
                    [1., 1.],
                    [1., 0.],
                    [0., 1.],
                    [0., 0.],
                ],
                [
                    [4.],
                    [3.],
                    [3.],
                    [2.],
                ]
            ))
            prediction = model.feed_through([5., 3.])
            self.assertAlmostEqual(prediction[0][0], 10, delta=1)

    def test_model_decorator(self):
        model = Model([2, 1], 0.001, 2000)
        try:
            model.feed_through([1., 1.])
            self.assertTrue(False)
        except RuntimeError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
