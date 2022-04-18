import unittest
from MAAT_STOCK_OPTIMISATION_SYSTEM import MAAT_STOCK_OPTIMISATION_SYSTEM


class TestMAAT_STOCK_OPTIMISATION_SYSTEM(unittest.TestCase):
    async def _start_app(self):
        App = MAAT_STOCK_OPTIMISATION_SYSTEM()
        App.run()

    def test_run(self):
        App = MAAT_STOCK_OPTIMISATION_SYSTEM()
        self._start_app()

    def test_knapsack_algorithm(self):
        weight_limit = 10
        weights= [1, 5, 9, 0]
        values = [10, 50, 60, 40]

        App = MAAT_STOCK_OPTIMISATION_SYSTEM().knapsack_algorithm(weight_limit, weights, values, 4)
        self.assertTrue(App,True)
    def test_portfolio(self):
        App=MAAT_STOCK_OPTIMISATION_SYSTEM()
        self._start_app()
        App.entry2.insert(0,"FB")
        k=App.portfolio()
        self.assertTrue(k,True)
        self._start_app()


if __name__ == "__main__":
    unittest.main()
