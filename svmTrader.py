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
    context.i = 0
    context.asset = symbol('btc_usd')
    context.base_price = None
    context.stakeInMarket = 0.0
    context.model = joblib.load('SVM_Model.pkl')
    context.tradeWindow = 1

    context.downTrend = False
    context.lastPosition = 0

    context.crossLow = False
    context.crossHigh = False
    context.neutral = True


def handle_data(context, data):
    time_frame = 60

    context.i += 1

    if context.i % 5 != 0:
        return

    # Skip as many bars as long_window to properly compute the average
    if context.i < time_frame * 4:
        return

    price = data.current(context.asset, 'price')
    close = data.history(context.asset, 'close', bar_count=120, frequency='5T')
    low = data.history(context.asset, 'low', bar_count=120, frequency='5T')
    high = data.history(context.asset, 'high', bar_count=120, frequency='5T')
    volume = data.history(context.asset, 'volume', bar_count=120, frequency='5T')

    rsi_s = ta.momentum.rsi(close, n=9)
    tsi_s = ta.momentum.tsi(close, r=14, s=9)

    rsi_l = ta.momentum.rsi(close, n=14)
    tsi_l = ta.momentum.tsi(close, r=25, s=13)

    mfi = ta.momentum.money_flow_index(high, low, close, volume, n=14)

    stochSig = ta.momentum.stoch_signal(high, low, close, n=14, d_n=3, fillna=False)

    # If base_price is not set, we use the current value. This is the
    # price at the first bar which we reference to calculate price_change.
    if context.base_price is None:
        context.base_price = price

    price_change = (price - context.base_price) / context.base_price
    cash = context.portfolio.cash

    if context.i % 360 == 0:
        print((context.i / 1440), "Days passed.")
        #print("Tsi value:", tsi_long[-1])
        #print("EMA_Bol value is:", tsiEMA_HBol[-1])

    # Save values for later inspection
    record(price=price,
           volume=volume,
           cash=cash,
           price_change=price_change,
           rsi_s=rsi_s[-1],
           rsi_l=rsi_l[-1],
           tsi_s=tsi_s[-1],
           tsi_l=tsi_l[-1],
           mfi=mfi[-1],
           stochSig=stochSig[-1]
           )

    # Since we are using limit orders, some orders may not execute immediately
    # we wait until all orders are executed before considering more trades.
    orders = context.blotter.open_orders
    if len(orders) > 0:
        return

    # Exit if we cannot trade
    if not data.can_trade(context.asset):
        return

    pos_amount = context.portfolio.positions[context.asset].amount

    totalData = functions.splitAndCompress_noPrice(rsi_s[60:], tsi_s[60:], rsi_l[60:], tsi_l[60:],
                                                   mfi[60:], stochSig[60:])

    prediction = context.model.predict(totalData[-1:])
    if context.i % 360 == 0:
        print(totalData[-1:])
        print("Prediction:", prediction[0])
        print()

    if prediction[0] == 1 and pos_amount < 0:
        order_target_percent(context.asset, 1)
        print(pos_amount)

    if prediction[0] == -1 and pos_amount > 0:
        order_target_percent(context.asset, 0)
        print(pos_amount)



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

    """
    # Fourth chart: Plot TSI
    ax4 = plt.subplot(514, sharex=ax1)
    perf.loc[:, ['tsi_long', 'tsiEMA_HighBol', 'tsiEMA_LowBol']].plot(ax=ax4, label="tsi_long")
    ax4.set_ylabel('TSI')
    # ax4.axhline(context.TSI_OverBought, color='darkgoldenrod')
    # ax4.axhline(context.TSI_OverSold, color='darkgoldenrod')
    start, end = ax4.get_ylim()
    ax4.yaxis.set_ticks(np.arange(-36, end, end / 5))
    ax4.axhline(36, color='darkgoldenrod')
    ax4.axhline(-20, color='darkgoldenrod')
    # Fifth Chart
    # ax5 = plt.subplot(515, sharex=ax1)
    # perf.loc[:, 'tsi'].plot(ax=ax4, label="tsi")
    # ax5.set_ylabel("TSI")
    # start, end = ax5.get_ylim()
    # ax5.yaxis.set_ticks(np.arange(0, end, end / 5))
  

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(bottom=0.2)
    start, end = ax1.get_ylim()

    perf.loc[:, 'price'].plot(
        ax=ax1,
        label='Price')
    # ax2.legend_.remove()
    ax1.set_ylabel('{asset}\n({base})'.format(
        asset=context.asset.symbol,
        base=base_currency
    ))

    perf.loc[:, ['tsi_long']].plot(ax=ax2, label="tsi_long")
    perf.loc[:, ['tsiEMA_LowBol']].plot(ax=ax2, label="tsiEMA_LowBol")
    perf.loc[:, ['tsiEMA_HighBol']].plot(ax=ax2, label="tsiEMA_HighBol")

    # perf.loc[:, ['rsi_short']].plot(ax=ax3, label="rsi_short")
    # perf.loc[:, ['rsi_long']].plot(ax=ax3, label="rsi_long")
    ax2.axhline(0, color='green')
    ax2.yaxis.set_ticks(np.arange(-30, 45, 5))

    ax2.legend_.remove()
    """

    plt.show()


if __name__ == '__main__':
    run_algorithm(
        capital_base=10000,
        data_frequency='minute',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='bitfinex',
        algo_namespace=NAMESPACE,
        quote_currency='usd',
        start=pd.to_datetime('2018-02-01', utc=True),
        end=pd.to_datetime('2018-02-28', utc=True),
    )
