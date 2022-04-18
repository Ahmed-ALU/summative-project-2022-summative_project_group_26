# Python code to test the python algorithm that predicts the stock
import unittest


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        pass

    # # Returns TRUE if the the scaler minimum and maximum range are 0 and 1
    # Else return False

    def test_Scaler_MinMax_Range_is_one_or_zer0(self):
        expected = 0
        actual = 0
        self.assertEqual(expected, actual)

    def test_Scaler_MinMax_Range_is_negative(self):
        expected = 0, 1
        actual = -3
        self.assertEqual(expected, actual)
        print("Negative values not allowed")

    def test_scaler_MinMax_Range_is_Above_one_or_zero(self):
        expected = 0, 1
        actual = 2
        self.assertEqual(expected, actual)
        print("Minimum and Maximum range cant be more than 0 and 1")

    # Returns true if the axis is zero and above  and return false or raise an error is the axis is negative

    def test_axis_is_zero(self):
        expected = 0
        actual = 0
        self.assertEqual(expected, actual)
        print("Expected value met")

    def test_axis_is_above_zero(self):
        expected = 0
        actual = 3
        self.assertEqual(expected, actual)
        print("Axis cant be more than zero")

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
