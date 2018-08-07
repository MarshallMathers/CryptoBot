import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
from logbook import Logger
from sklearn.externals import joblib
import functions
from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, )
from catalyst.exchange.utils.stats_utils import extract_transactions

console_width = 320
pd.set_option('display.width', console_width)
np.set_printoptions(linewidth=console_width)

NAMESPACE = 'SVM_Trader'
log = Logger(NAMESPACE)


def initialize(context):
    # Context.i is used to keep track of time, it is used in a variety of ways
    # including as a way to skip frames, keep our trader on pace, and print out updates
    # how far along our trader is.
    context.i = 0

    # Context.asset is used to reference price values
    context.asset = symbol('btc_usd')
    context.base_price = None

    # Keeps track of investments
    context.stakeInMarket = 0.0

    # Context.model loads in the pre-trained SVM model that will be making the predictions
    context.model = joblib.load('SVM_Model_svm5.pkl')
    context.tradeWindow = 1


def handle_data(context, data):
    # Sets the window size
    time_frame = 60

    # This is the counter that keeps track of time
    context.i += 1

    # Because the SVM is trained on 5 minute chunks, we only need to calculate
    # price every 5 minutes
    if context.i % 5 != 0:
        return

    # Skips 4 hours of data to ensure that when it starts collecting data to calculate
    # all of the TA signals there is plenty of data to read in.
    if context.i < time_frame * 4:
        return

    # This sets price to one value which is the current price.
    price = data.current(context.asset, 'price')

    # These take in the last 4 hours of 5 minute data
    close = data.history(context.asset, 'close', bar_count=48, frequency='5T')
    low = data.history(context.asset, 'low', bar_count=48, frequency='5T')
    high = data.history(context.asset, 'high', bar_count=48, frequency='5T')
    volume = data.history(context.asset, 'volume', bar_count=48, frequency='5T')

    #########################################################
    ####               TA Signals                        ####
    #########################################################

    rsi_s = ta.momentum.rsi(close, n=9)
    tsi_s = ta.momentum.tsi(close, r=12, s=9)

    rsi_m = ta.momentum.rsi(close, n=12)
    tsi_m = ta.momentum.tsi(close, r=15, s=9)

    rsi_m2 = ta.momentum.rsi(close, n=15)
    tsi_m2 = ta.momentum.tsi(close, r=18, s=12)

    rsi_l = ta.momentum.rsi(close, n=24)
    tsi_l = ta.momentum.tsi(close, r=27, s=18)

    rsi_el = ta.momentum.rsi(close, n=30)
    tsi_el = ta.momentum.tsi(close, r=35, s=25)

    mfi_s = ta.momentum.money_flow_index(high, low, close, volume, n=9)
    mfi_m = ta.momentum.money_flow_index(high, low, close, volume, n=15)
    mfi_l = ta.momentum.money_flow_index(high, low, close, volume, n=24)

    bband_h_s = ta.volatility.bollinger_hband_indicator(close, n=12)
    bband_h_m = ta.volatility.bollinger_hband_indicator(close, n=18)
    bband_h_l = ta.volatility.bollinger_hband_indicator(close, n=26)

    bband_l_s = ta.volatility.bollinger_lband_indicator(close, n=12)
    bband_l_m = ta.volatility.bollinger_lband_indicator(close, n=18)
    bband_l_l = ta.volatility.bollinger_lband_indicator(close, n=26)

    # If base_price is not set, we use the current value. This is the
    # price at the first bar which we reference to calculate price_change.
    if context.base_price is None:
        context.base_price = price

    # Calculate the price chance since the last 5 minutes
    price_change = (price - context.base_price) / context.base_price
    cash = context.portfolio.cash

    # This helps keep track of how much time has passed since the beginning of back testing
    # Prints the amount of time elapsed every 6 hours
    if context.i % 360 == 0:
        print((context.i / 1440), "Days passed.")

    # Save values for later inspection
    record(price=price,
           volume=volume,
           cash=cash,
           price_change=price_change,
           rsi_s=rsi_s[-1],
           rsi_m=rsi_m[-1],
           rsi_m2=rsi_m2[-1],
           rsi_l=rsi_l[-1],
           rsi_el=rsi_el[-1],
           tsi_s=tsi_s[-1],
           tsi_m=tsi_m[-1],
           tsi_m2=tsi_m2[-1],
           tsi_l=tsi_l[-1],
           tsi_el=tsi_el[-1],
           mfi_s=mfi_s[-1],
           bband_h=bband_h_s[-1],
           bband_l=bband_l_s[-1],
           )

    # Since we are using limit orders, some orders may not execute immediately
    # we wait until all orders are executed before considering more trades.
    orders = context.blotter.open_orders
    if len(orders) > 0:
        return

    # Exit if we cannot trade
    if not data.can_trade(context.asset):
        return

    # Load current positions into a variable
    pos_amount = context.portfolio.positions[context.asset].amount

    # Call the splitAndCompress function so that we can pass the current data to the SVM to give a prediction
    totalData = functions.splitAndCompress_Massive(rsi_s[-120:], rsi_m[-120:], rsi_m2[-120:], rsi_l[-120:], rsi_el[-120:],
                                                   tsi_s[-120:], tsi_m[-120:], tsi_m2[-120:], tsi_l[-120:], tsi_el[-120:],
                                                   mfi_s[-120:], mfi_m[-120:], mfi_l[-120:],
                                                   bband_l_s[-120:], bband_l_m[-120:], bband_l_l[-120:],
                                                   bband_h_s[-120:], bband_h_m[-120:], bband_h_l[-120:])

    # Call the prediction
    prediction = context.model.predict(totalData[-1:])

    # If 2 hours has elapsed since the last trade, check the prediction and trade accordingly
    if context.i % 120 == 0:
        # If the prediction is a buy, check that we aren't invested already
        if prediction[0] == 1 and pos_amount == 0:
            order_target_percent(context.asset, 1)
            print("Bought at:", price)
        # If the prediction is a sell, check if we are invested
        if prediction[0] == -1 and pos_amount > 0:
            order_target_percent(context.asset, 0)
            print("Sold at:", price)


def analyze(context, perf):
    # Get the base_currency that was passed as a parameter to the simulation
    exchange = list(context.exchanges.values())[0]
    quote_currency = exchange.quote_currency.upper()

    # First chart: Plot portfolio value using base_currency
    ax1 = plt.subplot(311)
    perf.loc[:, ['portfolio_value']].plot(ax=ax1)
    ax1.legend_.remove()
    ax1.set_ylabel('Portfolio Value\n({})'.format(quote_currency))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Second chart: Plot asset price, moving averages and buys/sells
    ax2 = plt.subplot(512, sharex=ax1)
    perf.loc[:, 'price'].plot(
        ax=ax2,
        label='Price')
    # ax2.legend_.remove()
    ax2.set_ylabel('{asset}\n({base})'.format(
        asset=context.asset.symbol,
        base=quote_currency
    ))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    transaction_df = extract_transactions(perf)
    if not transaction_df.empty:
        buy_df = transaction_df[transaction_df['amount'] > 0]
        sell_df = transaction_df[transaction_df['amount'] < 0]
        ax2.scatter(
            buy_df.index.to_pydatetime(),
            perf.loc[buy_df.index, 'price'],
            marker='^',
            s=100,
            c='green',
            label=''
        )
        ax2.scatter(
            sell_df.index.to_pydatetime(),
            perf.loc[sell_df.index, 'price'],
            marker='v',
            s=100,
            c='red',
            label=''
        )

    # Third chart: Compare percentage change between our portfolio
    # and the price of the asset
    ax3 = plt.subplot(513, sharex=ax1)
    perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax3)
    ax3.legend_.remove()
    ax3.set_ylabel('Percent Change')
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))


    plt.show()


# This is the main function that runs all of the def's.
if __name__ == '__main__':
    run_algorithm(
        # Starting portfolio is $10,000
        capital_base=10000,
        # We specify that we want minute data instead of daily data
        data_frequency='minute',
        # Call the initialize funtion that loads in our model
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        # Exchange the we are back testing on
        exchange_name='bitfinex',
        algo_namespace=NAMESPACE,
        # When printing and comparing portfolio values we want to be looking at USD
        quote_currency='usd',
        # The start of the back testing algorithm
        start=pd.to_datetime('2018-01-01', utc=True),
        # The end of the back testing algorithm
        end=pd.to_datetime('2018-03-18', utc=True),
    )
