import unittest
from octane.model import Model


# dummy test
class TestModel(unittest.TestCase):
    def test_model(self):
        model = Model([5, 10, 1], 0.001, 2000)
        with model:
            model.train((
                [
                    [2., 1., 0., 0., 0.],
                    [2., 0., 0., 0., 0.],
                    [2., 2., 0., 0., 0.],
                    [4., 0., 0., 1., 0.],
                ],
                [
                    [96.],
                    [99.],
                    [89.1],
                    [80.2],
                ]
            ))
            prediction = model.feed_through([2., 0., 0., 0., 0.])
            self.assertAlmostEqual(*prediction[0], 97, delta=5)
        self.assertEqual(True, True)

    def test_model_decorator(self):
        model = Model([2, 1], 0.001, 2000)
        try:
            model.feed_through([1., 1.])
            self.assertTrue(False)
        except RuntimeError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
