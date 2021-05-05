import unittest
import perceptron
from common import stepFunction
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class TestPerceptron(unittest.TestCase):
    def test_perceptron_simple(self):
        p = perceptron.Perceptron(stepFunction)
        data = [
            [1,5],
            [2,1],
            [3,2],
            [2.5,3],
            [-1,-2],
            [-1.5,-1],
            [-0.5,-3],
            [-3,-1],
            [0.5,-1]
        ]
        labels = [ 1, 1, 1, 1, -1, -1, -1, -1, -1 ]

        train_err = p.train(data, labels)
        print(f"Training error: {train_err}")
        
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        plt.scatter(x,y)

        def yOfX(x):
            return -(p.bias + p.weights[0]*x) / p.weights[1]

        line_x = min(x), max(x)
        line_y = [yOfX(an_x) for an_x in line_x]
        plt.plot(line_x, line_y)

        plt.savefig('out.png')

        for d,l in zip(data,labels):
            prediction = p(d)
            self.assertEqual(prediction, l)
        
        prediction = p([2,4])
        expected = 1
        self.assertEqual(prediction, expected)

if __name__ == "__main__":
    unittest.main()