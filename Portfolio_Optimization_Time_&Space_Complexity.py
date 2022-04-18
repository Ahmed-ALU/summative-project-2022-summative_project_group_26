# A python code to determine and space complexity of an algorithm
import timeit


class portfolio_optimization:
    def __init__(self):
        import timeit
        import matplotlib.pyplot as plt
        import yfinance as yf
        import pandas as pd
        from datetime import datetime
        self.stock_owning_limit = 2000
        self.stocks_and_limits = {'FB': 200, 'AMZN': 50, 'AAPL': 70, 'NFLX': 80, 'GOOG': 120}
        start_time = timeit.timeit()
        self.dummy_assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
        print(f"dummy assets:{self.dummy_assets}")

        self.assets = input(
            "Enter the symbols of the companies you want to invest in:\nYou may choose from list above\n").split()
        self.money_to_be_invested = int(input("Enter the money to invest\n"))
        stocksStartDate = '2013-01-01'
        today = datetime.today().strftime('%Y-%m-%d')
        self.all_stocks = pd.DataFrame()
        for symbol in self.assets:
            tmp_close = yf.download(symbol,
                                    start=stocksStartDate,
                                    end=today,
                                    progress=False)['Close']
            self.all_stocks = pd.concat([self.all_stocks, tmp_close], axis=1)

        self.all_stocks.columns = self.assets
        self.b = []
        end_time = timeit.timeit()
        time_taken = start_time - end_time
        print(time_taken)
        # plt.plot(time_taken)
        # plt.title(time_taken)
        # plt.show()

        #               TIME COMPLEXITY
        #     We have two independents for loops for the above function, where each runs O(N) times
        # but when we combine the two loops.
        # Time complexity of this one is constant O(N^2)
        # This is because the loop is a simple loop with infinite number of iterations
        #                SPACE COMPLEXITY
        # Here, each symbol in assets which is a string that takes up 20+(n/2)*4 bytes  and each c takes up 20+(n/2)*4
        # bytes
        # where n is the number of characters in  each string . The total space taken could be 20+(n/2)*4 +20+(n/2)*4
        # We ignore the other constants and
        # Therefore, the space complexity is O(N)

    def full_list_of_companies(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import csv
        dict_from_csv = {}
        start_time = timeit.timeit
        with open('Top+2000+Valued+Companies+with+Ticker+Symbols.csv', mode='r') as inp:
            reader = csv.reader(inp)
            dict_from_csv = {rows[0]: rows[1] for rows in reader}
        comprehensive_list_of_companies = []
        comprehensive_list_of_symbols = []
        end_time = timeit.timeit()
        time_taken = (start_time - end_time)
        print(time_taken)
        # plt.plot(time_taken)
        # plt.title(time_taken)
        # plt.show()
        # print(dict_from_csv)

        #                     TIME COMPLEXITY
        # determining the time complexity of this code.
        # When rows = 0, it will take 0 time to run the code and when rows is 1, it will run 1 time
        # Time complexity of this one is constant O(N)
        # This is because the loop is a simple loop with infinite number of iterations
        #                SPACE COMPLEXITY
        #

        start_time = timeit.timeit()
        for item, item2 in dict_from_csv.items():
            comprehensive_list_of_companies.append(item)
            comprehensive_list_of_symbols.append(item2)

        company_dict = {"companies": comprehensive_list_of_companies, "symbols": comprehensive_list_of_symbols}
        Visual_list_of_companies = pd.DataFrame(company_dict)
        print(Visual_list_of_companies)
        end_time = timeit.timeit()
        time_taken = start_time - end_time
        # plt.plot(time_taken)
        # plt.title(time_taken)
        # plt.show()

        #                     TIME COMPLEXITY
        # determining the time complexity of this code.
        # Time complexity of this one is constant O(N)
        # This is because the loop is a simple loop with infinite number of iterations
        #                SPACE COMPLEXITY
        # Here, length of item, item2 and variable i are used in the algorithm so,
        # the total space used is item * item2 + item(N) * s + 1 * s = 2N * s + s, where s is a unit space taken.
        # Therefore, the space complexity is O(N)
        #

    def stock_optimisation_trial_graph(self):
        import timeit
        import matplotlib.pyplot as plt
        title = 'portfolio Adj. Close Price History'
        start_time = timeit.timeit()
        for c in self.all_stocks.columns.values:
            plt.plot(self.all_stocks[c], label=c)
        plt.title(title)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Adj.price USD ($)', fontsize=20)
        plt.legend(self.all_stocks.columns.values, loc='upper left')
        plt.show()
        end_time = timeit.timeit()
        time_taken = start_time - end_time
        # plt.plot(time_taken)
        # plt.title(time_taken)
        # #plt.show()

        #               TIME COMPLEXITY
        #     We have two independents for loops for the above function, where each runs O(N) times
        # but when we combine the two loops.
        # Time complexity of this one is constant O(N^2)
        # This is because the loop is a simple loop with infinite number of iterations
        #                SPACE COMPLEXITY
        # Here, each symbol in assets which is a string takes up 20+(n/2)*4 bytes  and each c takes up 20+(n/2)*4 bytes
        # where n is the number of characters in  each string . The total space taken could be 20+(n/2)*4 +20+(n/2)*4
        # We ignore the other constants and
        # Therefore, the space complexity is O(N)

    def portfolio_optimisation(self):
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt import risk_models
        from pypfopt import expected_returns
        import pandas as pd

        # portfolio optimisation

        # Calculate the expected returns and the annualised covariance matrix of asset returns
        mu = expected_returns.mean_historical_return(self.all_stocks)
        print(mu)
        # s is the annualised covariance matrix
        s = risk_models.sample_cov(self.all_stocks)
        # efficient frontier is the set of optimal portfolios that offer the highest expected return for a defined
        # level of risk or the lowest risk for a given level of expected return
        ef = EfficientFrontier(mu, s)

        weights = ef.max_sharpe()
        # print(weights)

        cleaned_weights = ef.clean_weights()
        ef.portfolio_performance(verbose=True)

        from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
        latest_prices = get_latest_prices(self.all_stocks)

        allocation_of_funds = DiscreteAllocation(cleaned_weights, latest_prices,
                                                 total_portfolio_value=self.money_to_be_invested)
        allocation, leftover = allocation_of_funds.lp_portfolio()
        companies = []
        number_of_stocks = []
        for company, no_of_stocks in allocation.items():
            companies.append(company)
            number_of_stocks.append(no_of_stocks)
        optimized_assets = {"company": companies, "number of stocks": number_of_stocks}
        print("Below are is the optimized portfolio")

        print(pd.DataFrame(optimized_assets))
        print('Funds remaining: ${:.2f}'.format(leftover))

    def knapsack_implementation(self):
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt import risk_models
        from pypfopt import expected_returns

        from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
        weight_limits = []
        for item in self.assets:
            if item in self.stocks_and_limits.keys():
                weight_limits.append(self.stocks_and_limits[item])
        latest_prices = get_latest_prices(self.all_stocks)
        mu = expected_returns.mean_historical_return(self.all_stocks)
        s = risk_models.sample_cov(self.all_stocks)
        ef = EfficientFrontier(mu, s)

        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        allocation_of_funds = DiscreteAllocation(cleaned_weights, latest_prices,
                                                 total_portfolio_value=self.money_to_be_invested)
        allocation, leftover = allocation_of_funds.lp_portfolio()
        companies = []
        number_of_stocks = []
        for company, no_of_stocks in allocation.items():
            companies.append(company)
            number_of_stocks.append(no_of_stocks)
        optimized_assets = {"company": companies, "number of stocks": number_of_stocks}
        self.knapsack_algorithm(self.stock_owning_limit, weight_limits, latest_prices, len(self.assets))
        print(self.b)

    def knapsack_algorithm(self, W, wt, val, n):
        import timeit
        import matplotlib.pyplot as plt
        start_time = timeit.timeit()
        K = [[0 for w in range(W + 1)]
             for i in range(n + 1)]
        for i in range(n + 1):
            for w in range(W + 1):

                #                     TIME COMPLEXITY
                # determining the time complexity of this code.
                # the first two for loops run as independent loops. Therefore, the first loop runs (w+1)n times
                # and the other one runs (n+1)n times. If we combine the two, we would get, (w+1)n + (n+1)n.
                # So we ignore the constants (w+1) and (n+1). This give us O(n^2)
                # We also have a nested loop where the outer one runs O(n) times and the inner one that runs O(n^2)
                # an overall time complexity of the nested loop is O(n^2), if we would then combine all the loops
                # Therefore, time complexity of these loops is constant O(N^4)
                # This is because the loop is a simple loop with several number of iterations
                #                SPACE COMPLEXITY
                # Here, the space taken by the list K is equal to 4n bytes where n is the length of the list
                # so we have, size -- 4 bytes integer, w -- 4 bytes integer, i -- 4
                # bytes integer
                # The total space needed for the above algorithm to complete is 4n + 4 + 4 + 4 (bytes), but since
                # we have two independent loops and a nested loop, we can say the total space taken is going to be
                # O(n^2) + O(n^2)
                # The highest order of n in this equation is just n. Thus,
                # the space complexity of that operation is O(n^4)
                #
                if i == 0 or w == 0:
                    K[i][w] = 0

                elif wt[i - 1] <= w:
                    K[i][w] = max(val[i - 1]
                                  + K[i - 1][w - wt[i - 1]],
                                  K[i - 1][w])
                else:
                    K[i][w] = K[i - 1][w]

        max_value = K[n][W]
        print(max_value)
        w = W
        for i in range(n, 0, -1):
            if max_value <= 0:
                break
            if max_value == K[i - 1][w]:
                continue
            else:
                self.b.append(wt[i - 1])
                max_value = max_value - val[i - 1]
                w = w - wt[i - 1]
                end_time = timeit.timeit()
                time_taken = start_time - end_time
                plt.plot(time_taken)
                plt.title(time_taken)
                plt.show()


if '__main__' == __name__:
    sample_optimisation = portfolio_optimization()
    sample_optimisation.full_list_of_companies()
    sample_optimisation.stock_optimisation_trial_graph()
    sample_optimisation.portfolio_optimisation()
    sample_optimisation.knapsack_implementation()
