import dt
import unittest

class TestDT(unittest.TestCase):
    def test_dt_simple(self):
        tree = dt.DecisionTree()
        data = [
            ("Red", True, 3),
            ("Red", False, 1),
            ("Red", False, 3),
            ("Red", True, 1),
            ("Blue", False, 2),
            ("Blue", False, 3),
            ("Blue", True, 1),
            ("Blue", False, 1),
            ("Blue", True, 1),
            ("Blue", True, 1),
        ]
        labels = [ 1, 1, 0, 1, 0, 0, 1, 1, 0, 1 ]

        train_err = tree.train(data, labels)
        print(f"Training error: {train_err}")

        for d,l in zip(data,labels):
            prediction = tree(d)
            self.assertEqual(prediction, l)


if __name__ == "__main__":
    unittest.main()