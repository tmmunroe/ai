from common import manhattanDistance
import knn
import unittest

class TestKNN(unittest.TestCase):
    def test_knn_simple(self):
        nn = knn.KNN(1, manhattanDistance)
        data = [
            (1,0,0),
            (1,0,1),
            (1,1,1),
            (1,1,0),
            (-1,0,0),
            (-1,0,1),
            (-1,1,1),
            (-1,1,0)
        ]
        labels = [ 1, 1, 1, 1, 0, 0, 0, 0 ]

        train_err = nn.train(data, labels)
        print(f"Training error: {train_err}")

        for d,l in zip(data,labels):
            prediction = nn(d)
            self.assertEqual(prediction, l)

    def test_knn_three(self):
        nn = knn.KNN(3, manhattanDistance)
        data = [
            (1,0,0),
            (1,0,1),
            (1,1,1),
            (1,1,0),
            (-1,0,0),
            (-1,0,-1),
            (-1,-1,-1),
            (-1,-1,0)
        ]
        labels = [ 1, 1, 1, 1, 0, 0, 0, 0 ]

        train_err = nn.train(data, labels)
        print(f"Training error: {train_err}")

        test_data = [
            (2,2,2),
            (-2,-2,-2)
        ]
        expected_labels = [ 1,0 ]
        test_err = nn.test(test_data, expected_labels)
        self.assertEqual(test_err, 0)

        test_data = [
            (2,2,2),
            (-2,-2,-2)
        ]
        expected_labels = [ 0,1 ]
        test_err = nn.test(test_data, expected_labels)
        self.assertEqual(test_err, 2)

if __name__ == "__main__":
    unittest.main()