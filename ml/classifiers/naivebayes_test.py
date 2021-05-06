import naivebayes
import unittest

class TestNaiveBayes(unittest.TestCase):
    def test_nb_simple(self):
        print("test_nb_simple")
        nb = naivebayes.NaiveBayes()
        data = [
            "A",
            "B",
            "A",
            "A",
            "B",
            "B",
            "A",
            "A",
            "B",
            "B"
        ]
        labels = [ 
            "A",
            "B",
            "A",
            "A",
            "B",
            "B",
            "A",
            "A",
            "B",
            "B"
         ]

        train_err = nb.train(data, labels)
        print(f"Training error: {train_err}")

        for d,l in zip(data,labels):
            prediction = nb(d)
            self.assertEqual(prediction, l)

    def test_nb_simple_single_error(self):
        print("test_nb_simple_single_error")
        nb = naivebayes.NaiveBayes()
        data = [
            "A",
            "B",
            "A",
            "A",
            "B",
            "B",
            "A",
            "A",
            "B",
            "B"
        ]
        labels = [ 
            "A",
            "B",
            "A",
            "A",
            "B",
            "B",
            "A",
            "A",
            "B",
            "A"
         ]

        train_err = nb.train(data, labels)
        print(f"Training error: {train_err}")
        self.assertEqual(1, train_err)

    def test_nb_multi(self):
        print("test_nb_multi")
        data = [
            ("Sweet", "Red", "Spherical"),
            ("Sweet", "Yellow", "Tubular"),
            ("Sweet", "Green", "Spherical"),
            ("Tart", "Green", "Spherical"),
            ("Sweet", "Yellow", "Tubular"),
            ("Tart", "Green", "Spherical")
        ]
        labels = [
            "Apple",
            "Banana",
            "Apple",
            "Apple",
            "Banana",
            "Grape"
        ]
        nb = naivebayes.NaiveBayes()
        train_err = nb.train(data, labels)
        print(f"Training error: {train_err}")

        sample = ("Sweet", "Red", "Spherical")
        label = "Apple"
        prediction = nb(sample)
        print(f"Predicted: {prediction}")
        self.assertEqual(label, prediction)

        sample = ("Sweet", "Yellow", "Tubular")
        label = "Banana"
        prediction = nb(sample)
        print(f"Predicted: {prediction}")
        self.assertEqual(label, prediction)

        print("New sample....")
        sample = ("Sweet", "", "Tubular")
        label = "Banana"
        prediction = nb(sample)
        print(f"Predicted: {prediction}")
        self.assertEqual(label, prediction)



if __name__ == "__main__":
    unittest.main()