import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

"""
Here are the columns:

Open - the price the stock opened at.

High - the highest price during the day

Low - the lowest price during the day

Close - the closing price on the trading day

Volume - how many shares were traded

Stock doesn't trade every day (there is no trading on weekends and holidays), so some dates are missing.
"""


def stock_price_prediction():

    viable_companies = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
    while True:
        #to make user's input resemble the options we have
        company = input("Enter the company you want to see tomorrow's price prediction: ").replace(" ", "").upper()
        if company not in viable_companies:
            print("Please choose a company among the following:")
            print(f'{viable_companies}')
        else:
            break
    # when to start the data
    start = dt.datetime(2013, 1, 1)
    end = dt.datetime(2020,1,1)
    data = web.DataReader(company, 'yahoo', start, end)
    #     data
    # Prepare data
    # scale down all the values we have so that they fit in between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))  # from the sklearn.preprocessing module
    # We are predicting the price after markets have closed
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # how many days do we want to base our prediction on. How many days do I want to look back to decide what the price is going to be for the next day
    prediction_days = 60

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
    #so we will use from 2020 to now since we used 2013 to 2020 to train
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
    plt.plot(actual_price, color='blue', label=f"Actual {company} Price")
    plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")



    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()

    # Predict next day

    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"predicted price: {prediction[0][0]}")


stock_price_prediction()