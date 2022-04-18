# Final-Jan-April-2022
**Data Structures and Algorithms

Peer Group 26

Summative Project Documentation**

**MEMBERS**
Samuel Wanza
Myra Lugwiri
Ednah Akoth
Ahmed Mohamed
Jeremiah Ater



**Application Name**
MAAT STOCK OPTIMISATION SYSTEM

**Application Description**
Maat Stock Optimisation System is a  platform that provides traders with a competitive edge in the stock market industry. It aims to help traders make better-informed trading decisions where they maximise their profits while considering the risks involved and achieving portfolio diversification. The system calculates the range of stocks to invest in and the maximum return the trader will acquire. It Identifies the assets that would have maximised return on investment while minimising the risks associated with stock trading.

**Requirements **

In order to utilise the Maat Stock Optimisation System on your laptop you will need to have the following installed in your laptop. We are working with the assumption that you have the python environment set up and have a working IDE.
Python libraries to install include:

        Tkinter 
        Tkinter.ttk
        Pandas
        Yfinance
        Numpy
        Sklearn.preprocessing
        TensorFlow
        Matplotlib
        Matplotlib.pyplot
        Keras
        Keras.models
        Keras.layers
        PIL
        pypfopt


**Instructions: How to use the program**



Ensure you have Python and an Integrated Development environment(IDE) installed.
Install all the necessary python libraries (listed above) before running the program.

**MAAT_STOCK_OPTIMIZATION_SYSTEM.py is the main file. This is the file you will need to start the application**

When you click on run on your IDE, a Graphical User interface (GUI) will appear.

The User Interface partitioned into three visible parts:

**Inputs:**
        Enter the Stock to invest
        Enter amount to invest
**Outputs**
        Portfolio performance
        Predicted prices
        Maximum profit
        Preferable Stocks
**Graphs**

Enter the tickers/symbols for the companies/assets you want to invest in one at a time, under the “ENTER THE STOCK TO INVEST” input field. The following is the list of assets currently registered on the platform:
        FB
        AMZN 
        AAPL 
        NFLX
        GOOG

Next, enter the amount of money you want to invest in under the “ENTER AMOUNT TO INVEST” field. Please make sure that your input under this field is an integer. 

Thereafter, click on the OPTIMISE button. 

**At this stage, please note that the prediction model will run for approximately 2-3 minutes**. This is because the model gets trained using the previous data related to the asset/company you have chosen in order to give the user a prediction that is well informed. **During this time, the GUI will stay without activity for about 2-3 minutes**

Results will be displayed in the specific Output areas. 

You may change the input values and try other values for as many times as you please

