import unittest

from src.metrics.metric import Metric


class DummyMetric(Metric):
    def calculate(self, repository_data):
        return repository_data

class TestMetricBase(unittest.TestCase):
    def test_abstract_method(self):
        with self.assertRaises(TypeError):
            Metric()

    def test_subclass_implements_calculate(self):
        dummy = DummyMetric()
        self.assertEqual(dummy.calculate(42), 42)

if __name__ == "__main__":
    unittest.main()
