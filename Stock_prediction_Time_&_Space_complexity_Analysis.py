#!/usr/bin/python3
import tkinter as tk
import tkinter.ttk as ttk

from pandas import DataFrame
from tkinter import NW

# from portfolio_optimisation3 import *
import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from PIL import Image, ImageTk

global bar1
global z
z = str()


class PygubuWorksCopyApp:
    def __init__(self, master=None):
        self.stocks_and_limits = {'FB': 200, 'AMZN': 50, 'AAPL': 70, 'NFLX': 80, 'GOOG': 120}
        self.stock_owning_limit = 2000
        self.b = []

        self.asset = []
        self.assets = []
        self.all_stocks = pd.DataFrame()

        # build ui
        # Initiating the top level

        self.toplevel1 = tk.Toplevel(master, container="false")
        self.toplevel1.configure(borderwidth="10", relief="flat", takefocus=True)
        self.toplevel1.overrideredirect("False")
        self.toplevel1.title("STOCK OPTIMISER")
        self.toplevel1.maxsize(1200, 600)
        self.toplevel1.minsize(1200, 600)

        # Main Title on the screen
        self.label9 = ttk.Label(self.toplevel1)
        self.label9.configure(
            background="#4B9CD3",
            font="{@HP Simplified Jpan} 24 {}",
            foreground="white",
            justify="center",
            # relief="sunken",
            # takefocus=True,
            padding="7 3",
            # width= "Y",
            text="MAAT STOCK/PORTFOLIO OPTIMISATION SYSTEM")
        # self.label9.grid(column="1", row="0")
        # self.label9.pack()
        self.label9.grid(column=0, row=0, columnspan=6)

        self.inp = ttk.Label(self.toplevel1)
        self.inp.configure(text="----------------------------INPUTs--------------------------------", font='Bold 15')
        # self.inp.place(rely=.1, relx= .05)
        self.inp.grid(columnspan=2, column=0, row=1)

        # self.dash = ttk.Label(self.toplevel1)
        # self.dash.configure(text="-----------------------------------------------------", font= 'Bold 10')
        # self.dash.place(rely=.1, relx= .01)

        self.label10 = ttk.Label(self.toplevel1)
        self.label10.configure(
            font="TkTextFont", justify="right", padding="0", takefocus=False)
        self.label10.configure(text="ENTER THE STOCK TO INVEST: ", width="30")
        # self.label10.place(rely=.19, relx= .05)
        self.label10.grid(column=0, row=2)
        self.label10.configure(font='30')
        # self.label10.grid(
        #     column="0",
        #     columnspan="3",
        #     padx="10",
        #     pady="30",
        #     row="3",
        #     rowspan="3",
        #     sticky="w",
        # )

        self.entry2 = ttk.Entry(self.toplevel1)
        self.entry2.grid(column=1, row=2)
        # self.entry2.place(rely=.19, relx= .3)

        self.label14 = ttk.Label(self.toplevel1)
        self.label14.configure(text="ENTER AMOUT TO INVEST: ")
        # self.label14.place(rely=.27, relx= .05)
        self.label14.configure(font='30')
        self.label14.grid(row=3, column=0)

        self.entry4 = ttk.Entry(self.toplevel1)
        self.entry4.grid(row=3, column=1)

        # self.entry4.place(rely=.27, relx= .3)

        self.inpG = ttk.Label(self.toplevel1)
        self.inpG.configure(
            text="-----------------------------------------------------------------Graphs---"
                 "-------------------------------------------------",
            font='Bold 15', justify='left')
        # self.inp.place(rely=.5, relx= .05)
        self.inpG.grid(row=4, column=0, columnspan=6)

        self.inpS = ttk.Label(self.toplevel1)
        self.inpS.configure(text="        ", font='Bold 15')
        # self.inp.place(rely=.5, relx= .05)
        self.inpS.grid(row=1, column=2, rowspan=3)

        # Output is here

        self.inp2 = ttk.Label(self.toplevel1)
        self.inp2.configure(text="----------------------------Outputs--------------------------------", font='Bold 15')
        self.inp2.grid(column=3, row=1, columnspan=3)
        # self.inp.place(rely=.1, relx= .5)

        self.label13 = ttk.Label(self.toplevel1)
        self.label13.configure(text="PORTFOLIO \nPERFOMANCE", justify='center', font='Normal 11')
        # self.label13.place(rely=.15, relx= .5, width=120)
        self.label13.grid(column=3, row=2)
        # self.label13.grid(column="2", padx="10", pady="30", row="5", sticky="e")

        self.message1 = tk.Message(self.toplevel1)
        self.message1.configure(relief="sunken", takefocus=True, background='white')
        self.message1.grid(column=3, row=3)
        # self.message1.place(rely=.25, relx= .5, width=120, height=100)
        # self.message1.grid(column="2", row="6", sticky="e")

        self.label18 = ttk.Label(self.toplevel1)
        self.label18.configure(text="PREDICTED \nPRICES", justify='center', font='Normal 11')
        self.label18.grid(column=4, row=2)
        # self.label18.place(rely=.15, relx= .66, width=120)
        # self.label18.grid(column="0", padx="20", row="5", sticky="e")

        self.entry6 = tk.Entry(self.toplevel1)
        self.entry6.configure(relief="sunken", takefocus=True, background='white')
        self.entry6.grid(column=4, row=3)
        # self.entry5.place(rely=.28, relx= .65, width=120, height=30)
        # self.message1.grid(column="2", row="6", sticky="e")

        self.label20 = ttk.Label(self.toplevel1)
        self.label20.configure(text="MAXIMUM \nPROFIT", justify='center', font='Normal 11')
        self.label20.grid(column=5, row=2)
        # self.label20.place(rely=.15, relx= .82, width=120)
        # self.label20.grid(column="0", padx="20", row="5", sticky="e")

        self.entry8 = tk.Entry(self.toplevel1)
        self.entry8.configure(relief="sunken", takefocus=True, background='white')
        self.entry8.grid(column=5, row=3)
        # self.entry8.place(rely=.28, relx= .80, width=120, height=30)
        # self.entry8.grid(column="2", row="6", sticky="e")

        self.button1 = ttk.Button(self.toplevel1)
        self.button1.configure(text="OPTIMISE", command=self.portfolio)
        # self.button1.configure(text="OPTIMISE")
        self.button1.grid(column=5, row=6, columnspan=6)
        # self.button1.grid(column="0", pady="10", row="7")

        self.label15 = ttk.Label(self.toplevel1)
        self.label15.configure(text="STOCK PRICES SINCE 2013", font='Normal 12')
        # self.label15.place(rely=.55, relx= .15, width=290)
        # self.label15.grid(column="0", padx="10", row="8", sticky="w")
        self.label15.grid(column=0, row=5, columnspan=2)

        self.canvas2 = tk.Canvas(self.toplevel1)
        self.canvas2.configure(height="300", width="290", background='white')
        self.canvas2.grid(column=0, row=6, columnspan=2)
        # self.canvas2.place(rely=.6, relx= .1, width=290)
        # self.canvas2.grid(column="0", padx="10", pady="10", row="9", sticky="w")

        self.label16 = ttk.Label(self.toplevel1)
        self.label16.configure(relief="flat", text="STOCK PREDICTION", font='Normal 12')
        # self.label16.place(rely=.55, relx= .5, width=250)
        # self.label16.grid(column="0", row="8", sticky="e")
        self.label16.grid(column=2, row=5, columnspan=2)

        self.canvas3 = tk.Canvas(self.toplevel1)
        self.canvas3.configure(
            confine="false", height="300", relief="flat", width="290", background='white'
        )
        self.canvas3.grid(column=2, row=6, columnspan=2)

        # self.canvas3.place(rely=.6, relx= .45, width=290)
        # self.canvas3.grid(column="0", row="9", sticky="e")

        self.label19 = ttk.Label(self.toplevel1)
        self.label19.configure(text="PREFERABLE STOCKS", font='Normal 11')
        # self.label19.place(rely=.55, relx= .8, width=170)
        # self.label19.grid(column="1", row="8", sticky="n")
        self.label19.grid(column=5, row=4)

        self.entry7 = tk.Entry(self.toplevel1)
        self.entry7.configure(relief="sunken", takefocus=True, background='white')
        self.entry7.grid(column=5, row=5)
        # self.entry7.place(rely=.6, relx= .8, width=170, height=30)

        # Main widget
        self.mainwindow = self.toplevel1

    def run(self):
        self.mainwindow.mainloop()

    def portfolio(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        self.assets.append(self.entry2.get())
        # print(self.assets)

        # stocksStartDate = '2013-01-01'
        # today = datetime.today().strftime('%Y-%m-%d')
        # self.all_stocks = pd.DataFrame()

        stocksStartDate = '2013-01-01'
        today = datetime.today().strftime('%Y-%m-%d')
        # self.all_stocks = pd.DataFrame()
        v = set(self.assets)
        self.assets = list(v)

        for symbol in self.assets:

            if symbol != "" and symbol not in self.asset:
                self.asset.append(symbol)

                tmp_close = yf.download(symbol, start=stocksStartDate, end=today, progress=False)['Close']
                self.all_stocks = pd.concat([self.all_stocks, tmp_close], axis=1)

        # print(self.all_stocks)
        self.all_stocks.columns = self.asset
        # print(self.all_stocks)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # matplotlib.use('TkAgg')
        title = 'portfolio Adj. Close Price History'
        # fig, ax = plt.subplots()
        # plt.switch_backend('TkAgg')

        for c in self.all_stocks.columns.values:
            # plt.switch_backend('TkAgg')
            plt.plot(self.all_stocks[c], label=c)
        figure1 = plt.Figure(figsize=(2.8, 3.2), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, self.canvas2)
        bar1.get_tk_widget().grid(column="0", padx="10", pady="10", row="9", sticky="w")
        # df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
        self.all_stocks.plot(kind='line', legend=True, ax=ax1)
        ax1.set_title('portfolio Adj. Close Price History')

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

        #
        # plt.title(title)
        # plt.xlabel('Date', fontsize=20)
        # plt.ylabel('Adj.price USD ($)', fontsize=20)
        # plt.legend(self.all_stocks.columns.values, loc='upper left')
        # v=plt.show()
        # #img = ImageTk.PhotoImage(Image.open(v))
        # #self.canvas3.create_image(10, 10, anchor=NW, image=img)
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pandas_datareader as web
        import datetime as dt

        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, LSTM

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
        # when to start the data
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

        #               TIME COMPLEXITY
        # Time complexity of this one is constant O(N)
        # This is because the loop is a simple loop with infinite number of iterations
        #                SPACE COMPLEXITY
        # Here, the space taken by the list is equal to 4n bytes where n is the length of the
        # so we have, size -- 4 bytes integer, x_train_list -- 4 bytes integer, y_train_list -- 4 bytes integer
        # The total space needed for the above algorithm to complete is 4n + 4 + 4 + 4 (bytes)
        # The highest order of n in this equation is just n. Thus,
        # the space complexity of that operation is O(n)


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
        # plt.plot(actual_price, color='blue', label=f"Actual {company} Price")
        # plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
        #
        # plt.title(f"{company} Share Price")
        # plt.xlabel('Time')
        # plt.ylabel(f'{company} Share Price')
        # plt.legend()
        # plt.show()
        g = ["actual price", "predicted prices"]
        # df2=pd.DataFrame()
        data = {}
        datas = {"actual_price": actual_price, "predicted_prices": predicted_prices}
        df2 = DataFrame(datas, columns=["actual_price", "Predicted_prices"])
        ##
        # df2.columns=g
        # for c in df2.columns.values:
        #         # plt.switch_backend('TkAgg')
        #     plt.plot(df2[c], label=c)
        #

        figure2 = plt.Figure(figsize=(2.8, 3.2), dpi=100)
        ax2 = figure2.add_subplot(111)
        bar2 = FigureCanvasTkAgg(figure2, self.canvas3)
        bar2.get_tk_widget().grid(column=0, row=6, columnspan=2, sticky="e")
        df2.plot(kind='line', legend=True, ax=ax2)
        ax2.set_title('share price')

        # Predict next day

        real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        self.entry6.insert(0, prediction[0][0])

        import streamlit as st
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt import risk_models
        from pypfopt import expected_returns
        import pandas as pd

        # portfolio optimisation

        # Calculate the expected returns and the annualised covariance matrix of asset returns
        mu = expected_returns.mean_historical_return(self.all_stocks)
        # print(mu)
        # s is the annualised covariance matrix
        s = risk_models.sample_cov(self.all_stocks)
        # efficient frontier is the set of optimal portfolios that offer the highest expected return for a defined
        # level of risk or the lowest risk for a given level of expected return
        ef = EfficientFrontier(mu, s)

        weights = ef.max_sharpe()
        # print(weights)

        cleaned_weights = ef.clean_weights()
        ef.portfolio_performance(verbose=True)

        E = "Expected Return:    {:.2f}%\n".format(ef.portfolio_performance(verbose=True)[0] * 100)
        A = "Annual Volatility:   {:.2f}\n%".format(ef.portfolio_performance(verbose=True)[1] * 100)
        S = "Sharpe Ratio:    {:.2f}\n".format(ef.portfolio_performance(verbose=True)[2])

        # st.dataframe(ef.portfolio_performance(verbose=True))

        from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
        latest_prices = get_latest_prices(self.all_stocks)

        allocation_of_funds = DiscreteAllocation(cleaned_weights, latest_prices,
                                                 total_portfolio_value=int(self.entry4.get()))
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

    def knapsack_algorithm(self, W, wt, val, n):
        K = [[0 for w in range(W + 1)]
             for i in range(n + 1)]
        for i in range(n + 1):
            for w in range(W + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0

                elif wt[i - 1] <= w:
                    K[i][w] = max(val[i - 1]
                                  + K[i - 1][w - wt[i - 1]],
                                  K[i - 1][w])
                else:
                    K[i][w] = K[i - 1][w]

                    #                     TIME COMPLEXITY
                    # determining the time complexity of this code.
                    # the first two for loops run as independent loops. Therefore, the first loop runs (w+1)n times
                    # and the other one runs (n+1)n times. If we combine the two, we would get, (w+1)n + (n+1)n.
                    # So we ignore the constants (w+1) and (n+1). This give us O(n)
                    # We also have a nested loop where the outer one runs O(n) times and the inner one that runs O(n)
                    # an overall time complexity of the nested loop is O(n^2), if we would then combine all the loops
                    # Therefore, time complexity of these loops is constant O(N^2)
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

        max_value = K[n][W]
        self.entry8.insert(0, max_value)
        # print("p:",max_value)
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

        # return max_value


if __name__ == "__main__":
    app = PygubuWorksCopyApp()
    app.run()
