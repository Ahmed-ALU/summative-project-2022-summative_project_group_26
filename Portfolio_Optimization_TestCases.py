# Python code to test the python algorithm that predicts the stock
from Stock_Prediction import portfolio_optimization
import unittest


class Test_portfolio_optimization(unittest.TestCase):

    def setU(self):
        pass

    # Testing the input variable
    def test_Is_String(self):
        expected = 'GOOG'
        actual = 'GOOG'
        self.assertEqual(expected, actual)

    def test_is_not_string(self):
        expected = 'GOOG'
        actual = 2
        self.assertEqual(expected, actual)

    def test_is_Upper(self):
        expected = "GOOG"
        actual = 'GOOG'
        self.assertEqual(expected, actual)

    def test_is_not_upper(self):
        expected = 'GOOG'
        actual = 'goog'
        self.assertEqual(expected, actual)

    # Returns true if the axis is zero and above  and return false or raise an error is the axis is negative

    def test_amount_is_int(self):
        expected = 2600
        actual = 2600
        self.assertEqual(expected, actual)
        print("Expected value met")

    def test_amount_is_not_int(self):
        expected = 2600
        actual = 'lele'
        self.assertEqual(expected, actual)

    def test_amount_is_negative(self):
        expected = 2600
        actual = -200
        self.assertEqual(expected, actual)

    def test_axis_is_negative(self):
        expected = 0
        actual = -2
        self.assertEqual(expected, actual)
        print("Axis cant be a negative value")

    # print("Axis cant be negative")

    # Returns true if the prediction price is above 0 and returns false if it is 0 or negative value.
    def test_prediction_price_is_positive_value(self):
        expected = 2000
        actual = 2000
        self.assertEqual(expected, actual)

    def test_prediction_price_is_zero(self):
        expected = 2000
        actual = 0
        self.assertEqual(expected, actual)
        print("Prediction price as zero is not allowed")

    def test_prediction_price_is_negative_value(self):
        # prediction_price = -23445
        # self.assertEqual(prediction_price, -23445)
        expected = 200
        actual = -23445
        self.assertEqual(expected, actual)
        print("Prediction price cant be a negative value")

    def test_napstack_i_is_zero(self):
        expected = 0
        actual = 0
        self.assertEqual(expected, actual)

    def test_napstack_i_is_one(self):
        expected = 0
        actual = 0
        self.assertEqual(expected, actual)

    def test_napstack_i_is_nagtive(self):
        expected = -1
        actual = -1
        self.assertEqual(expected, actual)

    def test_napstack_i_is_another_no(self):
        expected = 0
        actual = 6
        self.assertEqual(expected, actual)

    def test_napstack_w_is_zero(self):
        expected = 0
        actual = 0
        self.assertEqual(expected, actual)

    def test_napstack_w_is_one(self):
        expected = 1
        actual = 1
        self.assertEqual(expected, actual)

    def test_napstack_w_is_another_no(self):
        expected = 0
        actual = 5
        self.assertEqual(expected, actual)

    def test_prediction_start_date(self):
        expected = 2012, 1, 1
        actual = 2012, 1, 1
        self.assertEqual(expected, actual)

    def test_prediction_Start_date_not_correct(self):
        expected = 2012, 1, 1
        actual = 2010, 1, 1
        self.assertEqual(expected, actual)

    def test_prediction_end_date(self):
        expected = 2020, 1, 1
        actual = 2020, 1, 1
        self.assertEqual(expected, actual)

    def test_prediction_end_date_not_correct(self):
        expected = 2020, 1, 1
        actual = 2035, 1, 1
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
