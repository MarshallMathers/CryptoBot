import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
from logbook import Logger

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, )
from catalyst.exchange.utils.stats_utils import extract_transactions

NAMESPACE = 'true_strength'
log = Logger(NAMESPACE)



def initialize(context):
    context.i = 0
    context.asset = symbol('ltc_usd')
    context.base_price = None
    context.signalHigh = 0
    context.signalLow = 0
    context.stakeInMarket = 0.0
    context.TSI_OverBought = 26.5
    context.TSI_OverSold = -20.2

    context.CandleStick = '1H'


def handle_data(context, data):

    time_frame = 60
    # Skip as many bars as long_window to properly compute the average
    context.i += 1
    if context.i % 60 != 0:
        return

    if context.i < time_frame*360:
        return

    # if context.i % 240 == 0:
    #    print((context.i / 60), "hours passed.")

    if context.i % 360 == 0:
        print((context.i / 1440), "days passed.")
    # Compute moving averages calling data.history() for each
    # moving average with the appropriate parameters. We choose to use
    # minute bars for this simulation -> freq="1m"
    # Returns a pandas dataframe.

    close = data.history(context.asset, 'close', bar_count=int(146), frequency='60T')
    close2 = data.history(context.asset, 'close', bar_count=int(360), frequency='1H')
    # close3 = data.history(context.asset, 'close', bar_count=int(30), frequency='1D')
    price = data.current(context.asset, 'price')
    volume = data.current(context.asset, 'volume')

    # low = data.history(context.asset, 'low', bar_count=int(time_frame), frequency='1T')#context.CandleStick)
    # high = data.history(context.asset, 'high', bar_count=int(time_frame), frequency='1T')#context.CandleStick)
    #
    # close2 = []
    # low2=[]
    # high2=[]
    # currClose = close[0]
    # currHigh = close[0]
    # currLow = close[0]
    # for i in range(len(close)):
    #     if i % 60 == 59:
    #         close2.append(currClose)
    #         close2.append(currHigh)
    #         close2.append(currLow)
    #     else:
    #         currClose = close[i]
    #         if high[i] > currHigh:
    #             currHigh = high[i]
    #         if high[i] > currHigh:
    #             currLow = low[i]

    tsi_long = np.array(ta.momentum.tsi(pd.Series(close2), r=42, s=30))
    tsi_short = np.array(ta.momentum.tsi(pd.Series(close), r=25, s=22))

    # rsi_long = ta.momentum.rsi(pd.Series(close3), n=14)
    # rsi_short = ta.momentum.rsi(pd.Series(close3), n=7)

    # tsiEMA = ta.trend.ema_slow(tsi, n_slow=720)
    # rsi = ta.momentum.rsi(pd.Series(close), n=30)

    # If base_price is not set, we use the current value. This is the
    # price at the first bar which we reference to calculate price_change.
    if context.base_price is None:
        context.base_price = price

    price_change = (price - context.base_price) / context.base_price


    # Save values for later inspection
    record(price=price,
           volume=volume,
           cash=context.portfolio.cash,
           price_change=price_change,
           tsi_long=tsi_long[-1],
           tsi_short=tsi_short[-1],
           # rsi_long=rsi_long[-1],
           # rsi_short=rsi_short[-1]
           # tsiEMA=tsiEMA[-1]
           )


    # Since we are using limit orders, some orders may not execute immediately
    # we wait until all orders are executed before considering more trades.
    orders = context.blotter.open_orders
    if len(orders) > 0:
        return

    # Exit if we cannot trade
    if not data.can_trade(context.asset):
        return

    # We check what's our position on our portfolio and trade accordingly
    pos_amount = context.portfolio.positions[context.asset].amount

    # Trading logic
    """
    if price < context.signalLow and context.stakeInMarket < 1.0:
        order_target_percent(context.asset, (context.stakeInMarket + 0.5))
        print("Buy", (pos_amount * price), "amount of LTC")
        context.stakeInMarket += .5
    if price > context.signalHigh and pos_amount > 0:
        order_target_percent(context.asset, (context.stakeInMarket - .5))
        print("Sold", (pos_amount * price), "amound of LTC")
        context.stakeInMarket -= .5

    """

    # if short_mavg > long_mavg and pos_amount == 0:
    # we buy 100% of our portfolio for this asset
    #	order_target_percent(context.asset, 1)
    # elif short_mavg < long_mavg and pos_amount > 0:
    # we sell all our positions for this asset
    #	order_target_percent(context.asset, 0)


def analyze(context, perf):
    # Get the base_currency that was passed as a parameter to the simulation
    exchange = list(context.exchanges.values())[0]
    base_currency = exchange.base_currency.upper()

    """
    # First chart: Plot portfolio value using base_currency
    ax1 = plt.subplot(511)
    perf.loc[:, ['portfolio_value']].plot(ax=ax1)
    ax1.legend_.remove()
    ax1.set_ylabel('Portfolio Value\n({})'.format(base_currency))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Second chart: Plot asset price, moving averages and buys/sells
    ax2 = plt.subplot(512, sharex=ax1)
    perf.loc[:, 'price'].plot(
        ax=ax2,
        label='Price')
    #ax2.legend_.remove()
    ax2.set_ylabel('{asset}\n({base})'.format(
        asset=context.asset.symbol,
        base=base_currency
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

    # # Third chart: Compare percentage change between our portfolio
    # # and the price of the asset
    # ax3 = plt.subplot(513, sharex=ax1)
    # perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax3)
    # ax3.legend_.remove()
    # ax3.set_ylabel('Percent Change')
    # start, end = ax3.get_ylim()
    # ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Fourth chart: Plot our cash
    ax4 = plt.subplot(514, sharex=ax1)
    perf.loc[:, ['tsi_long']].plot(ax=ax4, label="tsi_long")
    perf.loc[:, ['tsi_short']].plot(ax=ax4, label="tsi_short")
    ax4.set_ylabel('TSI')
    #ax4.axhline(context.TSI_OverBought, color='darkgoldenrod')
    #ax4.axhline(context.TSI_OverSold, color='darkgoldenrod')
    start, end = ax4.get_ylim()
    ax4.yaxis.set_ticks(np.arange(0, end, end / 5))

    # Fifth Chart
    # ax5 = plt.subplot(515, sharex=ax1)
    # perf.loc[:, 'tsi'].plot(ax=ax4, label="tsi")
    # ax5.set_ylabel("TSI")
    # start, end = ax5.get_ylim()
    # ax5.yaxis.set_ticks(np.arange(0, end, end / 5))
    """

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
    perf.loc[:, ['tsi_short']].plot(ax=ax2, label="tsi_short")

    # perf.loc[:, ['rsi_short']].plot(ax=ax3, label="rsi_short")
    # perf.loc[:, ['rsi_long']].plot(ax=ax3, label="rsi_long")

    # ax2.yaxis.set_ticks(np.arange(0, end, end / 5))
    # ax2 = plt.subplot(512, sharex=ax1)

    # ax2.legend_.remove()



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
        base_currency='usd',
        start=pd.to_datetime('2017-06-01', utc=True),
        end=pd.to_datetime('2017-07-30', utc=True),
    )
