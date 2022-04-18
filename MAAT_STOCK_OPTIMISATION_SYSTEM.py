# We first import the tkinter module,a library will help with the user interface

try:
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import NW
    # We then import the pandas library , a module that will help us with data manipulation
    from pandas import DataFrame
    import pandas as pd
    # We then import yfinance, an api that helps obtain stock data from yahoo finance
    import yfinance as yf
    # We then import the datetime module to help us obtain real time data
    from datetime import datetime
    import datetime as dt
    # We then import matplotlib for plotting
    import matplotlib.pyplot as plt
    # We then import the FigureCanvasTkAgg that will display the graphs in the graphical user interface

    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
    # We then import the figure module that will help us display the graphs in the user interface
    from matplotlib.figure import Figure
    # We then import the image and imageTk module that help us display the graphs as images in the user interface
    from PIL import Image, ImageTk
    # We the import NumPy which will be used to perform a wide variety of mathematical operations on arrays in the
    # prediction algorithm
    import numpy as np
    # We then import pandas_datareader that will help us obtain data from yfinace
    import pandas_datareader as web
    # We then import The sklearn.preprocessing package that provides several common utility functions and transformer
    # classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
    from sklearn.preprocessing import MinMaxScaler
    # We then import tensorflow.keras Keras is a neural network Application Programming Interface (API) for Python
    # that is tightly integrated with TensorFlow, which is used to build machine learning models. Keras' models offer
    # a simple, user-friendly way to define a neural network, which will then be built for you by TensorFlow.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
except ModuleNotFoundError:
    print("Module not found. Please ensure all of the following are installed:"
          "1. numpy"
          "2. pandas_datareader"
          "3. sklearn.preprocessing"
          "4. tkinter"
          "5. tkinter.ttk"
          "6. pandas"
          "7. yfinance"
          "8. datetime"
          "9. matplotlib.pyplot"
          "10. matplotlib.backends.backend_tkagg"
          "11. tensorflow"
          "12. Keras"
          "13. keras.models"
          "14. keras.layers"
          "15. PIL")

# We then create a variable bar1 which stores the graph that has stock prices of the previous years
global bar1
# We then create a global variable z for getting the output to the user interface
global z
# The gui can only accept string ouputs, thus intergers need to be converted to strings before displaying
z = str()


# We then create a class MAAT_STOCK_OPTIMISATION_SYSTEM  that is a template for application and will contain all
# application methods
class MAAT_STOCK_OPTIMISATION_SYSTEM:
    # We then create a constructor for the class
    def __init__(self, master=None):
        # We first create a dictionary self.stocks_and_limits which has stocks that we shall use for our testing,
        # purposes The stocks are estimates
        self.stocks_and_limits = {'FB': 200, 'AMZN': 50, 'AAPL': 70, 'NFLX': 80, 'GOOG': 120}
        # We assume we are in a autocratic society thus there is limit to the number of stocks you can own
        self.stock_owning_limit = 2000
        # We then initialize a list self.b which will store all stocks after optimisation
        self.b = []
        # We then initialize an empty list self.assets for taking the stock inputs
        self.assets = []
        # We then initialize an empty list self.asset for removing duplicates and empty entries
        self.asset = []

        # We then create an instance of a dataframe
        self.all_stocks = pd.DataFrame()



        # We then create an instance of  Tkinter usinf the Toplevel model
        self.toplevel1 = tk.Toplevel(master, container="false")
        # We then create set the size of the main window
        self.toplevel1.configure(borderwidth="10", relief="flat", takefocus=True)
        self.toplevel1.overrideredirect("False")
        # Setting the pop_up title
        self.toplevel1.title("STOCK OPTIMISER")
        # Setting media queries for the main window
        self.toplevel1.maxsize(1200, 600)
        self.toplevel1.minsize(1200, 600)
        # Setting the label for the main title
        self.label9 = ttk.Label(self.toplevel1)
        self.label9.configure(
            background="#4B9CD3",
            font="{@HP Simplified Jpan} 24 {}",
            foreground="white",
            justify="center",
            padding="7 3",
            text="MAAT STOCK/PORTFOLIO OPTIMISATION SYSTEM")
        self.label9.grid(column=0, row=0, columnspan=6)

        # We then subdivide our main window to four various displays
        self.inp = ttk.Label(self.toplevel1)
        self.inp.configure(text="----------------------------INPUTs--------------------------------", font='Bold 15')
        # Setting  the size and position of the label
        self.inp.grid(columnspan=2, column=0, row=1)
        self.label10 = ttk.Label(self.toplevel1)
        # Setting the dimensions for the stocks to invest input
        self.label10.configure(
            font="TkTextFont", justify="right", padding="0", takefocus=False)
        self.label10.configure(text="ENTER THE STOCK TO INVEST: ", width="30")
        self.label10.grid(column=0, row=2)
        self.label10.configure(font='30')
        # Setting the input for the amount to invest
        self.entry2 = ttk.Entry(self.toplevel1)
        self.entry2.grid(column=1, row=2)

        self.label14 = ttk.Label(self.toplevel1)
        self.label14.configure(text="ENTER AMOUNT TO INVEST: ")
        self.label14.configure(font='30')
        self.label14.grid(row=3, column=0)
        # Entry for Amount To invest
        self.entry4 = ttk.Entry(self.toplevel1)
        self.entry4.grid(row=3, column=1)
        # Setting the location of the graph ouputs/displays
        self.inpG = ttk.Label(self.toplevel1)
        self.inpG.configure(
            text="-----------------------------------------------------------------Graphs----------------------------------------------------",
            font='Bold 15', justify='left')
        self.inpG.grid(row=4, column=0, columnspan=6)

        # The location for the ouputs in the entry widgets
        self.inpS = ttk.Label(self.toplevel1)
        self.inpS.configure(text="        ", font='Bold 15')
        self.inpS.grid(row=1, column=2, rowspan=3)

        # Output is here

        self.inp2 = ttk.Label(self.toplevel1)
        self.inp2.configure(text="----------------------------Outputs--------------------------------", font='Bold 15')
        self.inp2.grid(column=3, row=1, columnspan=3)
        # Setting the porfolio performance label

        self.label13 = ttk.Label(self.toplevel1)
        self.label13.configure(text="PORTFOLIO \nPERFOMANCE", justify='center', font='Normal 11')
        self.label13.grid(column=3, row=2)

        # This where the portfolio performance anlysis would display
        self.message1 = tk.Message(self.toplevel1)
        self.message1.configure(relief="sunken", takefocus=True, background='white')
        self.message1.grid(column=3, row=3)
        # This where we will display the predicted prices
        self.label18 = ttk.Label(self.toplevel1)
        self.label18.configure(text="PREDICTED \nPRICES", justify='center', font='Normal 11')
        self.label18.grid(column=4, row=2)

        # Setting the location of the maximum profit output
        self.entry6 = tk.Entry(self.toplevel1)
        self.entry6.configure(relief="sunken", takefocus=True, background='white')
        self.entry6.grid(column=4, row=3)

        self.label20 = ttk.Label(self.toplevel1)
        self.label20.configure(text="MAXIMUM \nPROFIT", justify='center', font='Normal 11')
        self.label20.grid(column=5, row=2)

        self.entry8 = tk.Entry(self.toplevel1)
        self.entry8.configure(relief="sunken", takefocus=True, background='white')
        self.entry8.grid(column=5, row=3)
        # Setting the optimise button.
        # On clicking this button the self.portfolio method runs which encompasses all the functionality of our system.
        self.button1 = ttk.Button(self.toplevel1)
        self.button1.configure(text="OPTIMISE", command=self.portfolio)
        self.button1.grid(column=5, row=6, columnspan=6)

        # Setting the stock prices analysis label
        self.label15 = ttk.Label(self.toplevel1)
        self.label15.configure(text="STOCK PRICES SINCE 2013", font='Normal 12')
        self.label15.grid(column=0, row=5, columnspan=2)

        self.canvas2 = tk.Canvas(self.toplevel1)
        self.canvas2.configure(height="300", width="290", background='white')
        self.canvas2.grid(column=0, row=6, columnspan=2)

        # Setting the stock prediction label and its dimensions and position
        self.label16 = ttk.Label(self.toplevel1)
        self.label16.configure(relief="flat", text="STOCK PREDICTION", font='Normal 12')
        self.label16.grid(column=2, row=5, columnspan=2)

        self.canvas3 = tk.Canvas(self.toplevel1)
        self.canvas3.configure(
            confine="false", height="300", relief="flat", width="290", background='white'
        )
        self.canvas3.grid(column=2, row=6, columnspan=2)
        # Setting the preferable stocks label , its position and various dimensions and the place the output will appear
        self.label19 = ttk.Label(self.toplevel1)
        self.label19.configure(text="PREFERABLE STOCKS", font='Normal 11')
        self.label19.grid(column=5, row=4)

        self.entry7 = tk.Entry(self.toplevel1)
        self.entry7.configure(relief="sunken", takefocus=True, background='white')
        self.entry7.grid(column=5, row=5)
        # Assigning the gui template to the self.mainWindow
        self.mainwindow = self.toplevel1

    def run(self):
        # We then run the mainloop method which starts our GUI
        self.mainwindow.mainloop()

    def portfolio(self):
        """
        In this method, we have integrated the various tasks that our application is supposed to perform.
        We have the stock prediction functionality, the portfolio optimisation functionality and stock prediction
        functionality
        We daresay, this the actually the program which our application revolves around

        """

        import pandas as pd
        import matplotlib.pyplot as plt
        self.entry2.get().replace(" ", "")  # removing the spaces in the user's input
        if self.entry2.get() == '':
            self.entry2.insert(0, "Please enter a valid asset: 'FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG' ")
        else:
            self.assets.append(self.entry2.get())

        stocksStartDate = '2013-01-01'
        today = datetime.today().strftime('%Y-%m-%d')
        # We perform set conversion to remove duplicates in self. assets
        v = set(self.assets)
        self.assets = list(v)

        # We are iterating through each symbol in the self.assets and downloading its data in yahoo finance
        for symbol in self.assets:

            if symbol != "" and symbol not in self.asset:
                self.asset.append(symbol)

                tmp_close = yf.download(symbol, start=stocksStartDate, end=today, progress=False)['Close']
                self.all_stocks = pd.concat([self.all_stocks, tmp_close], axis=1)

        #Setting th asset symbols to be the column heads of our dataframe
        self.all_stocks.columns = self.asset
        #setting the graph and its features
        try:
            for c in self.all_stocks.columns.values:
                plt.plot(self.all_stocks[c], label=c)
            figure1 = plt.Figure(figsize=(2.8, 3.2), dpi=100)
            ax1 = figure1.add_subplot(111)
            bar1 = FigureCanvasTkAgg(figure1, self.canvas2)
            bar1.get_tk_widget().grid(column="0", padx="10", pady="10", row="9", sticky="w")
            self.all_stocks.plot(kind='line', legend=True, ax=ax1)
            ax1.set_title('portfolio Adj. Close Price History')
        except TypeError:
            self.canvas2.configure(text="No Data to plot due to an error in input")

        viable_companies = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
        while True:
            # to make user's input resemble the options we have company = input("Enter the company you want to see
            # tomorrow's price prediction: ").replace(" ", "").upper()
            company = self.entry2.get()
            if company not in viable_companies:
                print("Please choose a company among the following:")
                print(f'{viable_companies}')
            else:
                break
        start = dt.datetime(2013, 1, 1)
        end = dt.datetime(2020, 1, 1)
        data = web.DataReader(company, 'yahoo', start, end)
        #     data
        # Prepare data
        # scale down all the values we have so that they fit in between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))  # from the sklearn.preprocessing module
        # We are predicting the price after markets have closed
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # how many days do we want to base our prediction on. How many days do I want to look back to decide what the
        # price is going to be for the next day
        prediction_days = 1

        x_train = []
        y_train = []

        # loading the data from the past 60 days into the training set
        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])
        # convert into numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # Build the model
        model = Sequential()

        # specify the layers
        # the more units you have the longer you will have to train the model.
        # also risk overfitting if too many layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))  # prediction of the next closing value

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32)

        '''Test the model accuracy on existing data'''

        # load some test data
        # has to be data that the model has not seen before
        # so we will use from 2020 to now since we used 2013 to 2020 to train
        test_start = dt.datetime(2020, 1, 1)
        test_end = dt.datetime.now()

        test_data = web.DataReader(company, 'yahoo', test_start, test_end)
        actual_price = test_data['Close'].values

        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

        model_inputs = model_inputs.reshape(-1, 1)
        # scale down using scaler we defined
        model_inputs = scaler.transform(model_inputs)

        # make predictions

        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices = model.predict(x_test)
        # Inverting the scaler
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # plot the test predictions
        try:
            g = ["actual price", "predicted prices"]
            datas = {"actual_price": actual_price, "predicted_prices": predicted_prices}
            df2 = DataFrame(datas, columns=["actual_price", "Predicted_prices"])

            figure2 = plt.Figure(figsize=(2.8, 3.2), dpi=100)
            ax2 = figure2.add_subplot(111)
            bar2 = FigureCanvasTkAgg(figure2, self.canvas3)
            bar2.get_tk_widget().grid(column=0, row=6, columnspan=2, sticky="e")
            df2.plot(kind='line', legend=True, ax=ax2)
            ax2.set_title('share price')
        except TypeError:
            self.canvas3.configure(text="No Data to plot")

        # Predict next day

        real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        self.entry6.insert(0, prediction[0][0])
        try:
            #The pypfopt module helps to perfrom portfolio optimisation ssing the already agreed trends and values in the
            #stock market

            from pypfopt.efficient_frontier import EfficientFrontier
            from pypfopt import risk_models
            from pypfopt import expected_returns
            import pandas as pd
        except ModuleNotFoundError:
            print('Module not found. Try installing all of the following modules:'
                  '1. pypfopt'
                  '2. pypfopt.efficient_frontier'
                  '3. pandas')

        # portfolio optimisation

        # Calculate the expected returns and the annualised covariance matrix of asset returns
        mu = expected_returns.mean_historical_return(self.all_stocks)
        # print(mu)
        # s is the annualised covariance matrix
        s = risk_models.sample_cov(self.all_stocks)
        # efficient frontier is the set of optimal portfolios that offer the highest expected return for a defined level of risk
        # or the lowest risk for a given level of expected return
        ef = EfficientFrontier(mu, s)

        weights = ef.max_sharpe()

        cleaned_weights = ef.clean_weights()
        #Displaying the analysed portfolio performance, verbose is true since its hard coded to only print values if specified
        ef.portfolio_performance(verbose=True)

        E = "Expected Return:    {:.2f}%\n".format(ef.portfolio_performance(verbose=True)[0] * 100)
        A = "Annual Volatility:   {:.2f}\n%".format(ef.portfolio_performance(verbose=True)[1] * 100)
        S = "Sharpe Ratio:    {:.2f}\n".format(ef.portfolio_performance(verbose=True)[2])

        from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
        latest_prices = get_latest_prices(self.all_stocks)

        try:
            entry_for_amount = int(self.entry4.get())
            allocation_of_funds = DiscreteAllocation(cleaned_weights, latest_prices,
                                                     total_portfolio_value=entry_for_amount)
            allocation, leftover = allocation_of_funds.lp_portfolio()

            companies = []
            number_of_stocks = []
            for company, no_of_stocks in allocation.items():
                companies.append(company)
                number_of_stocks.append(no_of_stocks)
            optimized_assets = {"company": companies, "number of stocks": number_of_stocks}
            z = E, A, S, pd.DataFrame(optimized_assets)
            self.message1.configure(text=z)
            weight_limits = []
            for item in self.asset:
                if item in self.stocks_and_limits.keys():
                    weight_limits.append(self.stocks_and_limits[item])
            self.knapsack_algorithm(self.stock_owning_limit, weight_limits, latest_prices, len(self.asset))
            self.entry7.insert(0, self.b)

        except ValueError:
            self.entry4.insert(0, "Please enter an integer")

        return True

    def knapsack_algorithm(self, owning_limit, weight_limit, val_prices, num_assests):
        """Returns the maximum value a user will get when they invest in specific stocks and the list of the maximum
                values for specific stocks they should invest in """

        # initializes a two-dimensional array with zero in it, a matrix to hold the possible maximum return user gets
        combinations = [[0 for w in range(owning_limit + 1)] for i in range(num_assests + 1)]
        # loops through list with the stock owning limits and the number of assets to fill the K two dimensional
        # array with possible maximum value combination
        for i in range(num_assests + 1):
            for w in range(owning_limit + 1):
                if i == 0 or w == 0:
                    combinations[i][w] = 0

                elif weight_limit[i - 1] <= w:
                    combinations[i][w] = max(val_prices[i - 1]
                                             + combinations[i - 1][w - weight_limit[i - 1]],
                                             combinations[i - 1][w])
                else:
                    combinations[i][w] = combinations[i - 1][w]

        # The maximum value a user will get within the restricted owning limits and number of assets
        max_value = combinations[num_assests][owning_limit]
        self.entry8.insert(0, max_value)
        w = owning_limit
        for i in range(num_assests, 0, -1):
            if max_value <= 0:
                break
            if max_value == combinations[i - 1][w]:  # check to see if the max_value identified is in the matrix
                # containing all possible combinations and they are equal
                continue
            else:
                self.b.append(
                    weight_limit[i - 1])  # Appends the weight limit an investor can invest in to get maximum returns
                max_value = max_value - val_prices[i - 1]
                w = w - weight_limit[i - 1]
        return True


if __name__ == "__main__":
    app = MAAT_STOCK_OPTIMISATION_SYSTEM()
    app.run()
