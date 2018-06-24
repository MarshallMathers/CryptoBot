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
    context.asset = symbol('eth_usd')
    context.base_price = None
    context.stakeInMarket = 0.0
    context.TSI_OverBought = 30
    context.TSI_OverSold = -19

    context.tradeWindow = 1

    context.downTrend = False
    context.lastPosition = 0

    context.crossLow = False
    context.crossHigh = False
    context.neutral = True

def handle_data(context, data):

    time_frame = 60

    context.i += 1

    if context.i % 60 != 0:
        return

    # Skip as many bars as long_window to properly compute the average
    if context.i < time_frame*528:
        return


    close = data.history(context.asset, 'close', bar_count=int(528), frequency='1H')
    price = data.current(context.asset, 'price')
    volume = data.current(context.asset, 'volume')

    # tsi_long = ta.momentum.tsi(pd.Series(close), r=55, s=35)
    # tsiEMA = ta.trend.ema_slow(pd.Series(tsi_long), n_slow=100)
    # tsiEMA_HBol = ta.volatility.bollinger_hband(pd.Series(tsiEMA), n=75, ndev=3)
    # tsiEMA_LBol = ta.volatility.bollinger_lband(pd.Series(tsiEMA), n=75, ndev=3)

    tsi_long = ta.momentum.tsi(pd.Series(close), r=30, s=35)
    tsiEMA = ta.trend.ema_slow(pd.Series(tsi_long), n_slow=100)
    tsiEMA_HBol = ta.volatility.bollinger_hband(pd.Series(tsiEMA), n=75, ndev=3)
    tsiEMA_LBol = ta.volatility.bollinger_lband(pd.Series(tsiEMA), n=75, ndev=3)


    # rsi_long = ta.momentum.rsi(pd.Series(close3), n=14)
    # rsi_short = ta.momentum.rsi(pd.Series(close3), n=7)

    # tsiEMA = ta.trend.ema_slow(tsi, n_slow=720)
    # rsi = ta.momentum.rsi(pd.Series(close), n=30)

    # If base_price is not set, we use the current value. This is the
    # price at the first bar which we reference to calculate price_change.
    if context.base_price is None:
        context.base_price = price

    price_change = (price - context.base_price) / context.base_price
    cash = context.portfolio.cash

    if context.i % 360 == 0:
        print((context.i / 1440), "Days passed.")
        print("Tsi value:", tsi_long[-1])
        print("EMA_Bol value is:", tsiEMA_HBol[-1])

    # Save values for later inspection
    record(price=price,
           volume=volume,
           cash=cash,
           price_change=price_change,
           tsi_long=tsi_long[-1],
           tsiEMA=tsiEMA[-1],
           tsiEMA_HighBol=tsiEMA_HBol[-1],
           tsiEMA_LowBol=tsiEMA_LBol[-1]
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



    # print( tsi_long[-1], "Tsi value")

    if (tsi_long[-1] > (tsiEMA_HBol[-1])*1.05) and (context.crossLow or context.neutral):
        context.crossHigh = True
        context.crossLow = False
        context.neutral = False
        # print("tsi long is crossing the high ")
    if tsi_long[-1] < (tsiEMA_LBol[-1]*0.95) and (context.crossHigh or context.neutral):
        context.crossHigh = False
        context.crossLow = True
        context.neutral = False
        # print("tsi long is crossing the low ")



    if not context.neutral:

        if context.crossHigh:
            if tsi_long[-1] < tsiEMA_HBol[-1] and pos_amount > 0:
                order_target_percent(context.asset, 0)
                context.crossHigh = False
                context.neutral = True
                # print("Selling?")
            if tsi_long[-1] > context.TSI_OverBought and pos_amount > 0\
                    and tsi_long[-1] < tsi_long[-2]:
                order_target_percent(context.asset, 0)



        elif context.crossLow:
            if tsi_long[-1] > (tsiEMA_LBol[-1]*1.8) and pos_amount < 1.0:
                order_target_percent(context.asset, 1)
                context.crossLow = False
                context.neutral = True
                # print("Buying?")






    # We check what's our position on our portfolio and trade accordingly

    # Trading logic
    """
    
    if context.downTrend:
        if tsi_long[-1] < context.TSI_OverSold and pos_amount < 1.0:
            order_target_percent(context.asset, 1)
            context.lastPosition = price
            context.downTrend = False

    else:
        if (((context.lastPosition * pos_amount)*.90) > (price * pos_amount) and pos_amount > 0):
            order_target_percent(context.asset, 0)
            context.downTrend = True

        if tsi_long[-1] > tsiEMA[-1] and pos_amount < 1.0:
            order_target_percent(context.asset, 1)
            context.lastPosition = price

        if tsi_long[-1] < tsiEMA[-1] and pos_amount > 0:
            order_target_percent(context.asset, 0)

    """



    """
    
    if tsi_long[-1] >= context.TSI_OverBought and pos_amount > 0:
        order_target_percent(context.asset, 0)
        print("Sold everything for $", (pos_amount * price))
        context.stakeInMarket = 0
        context.canTrade = False
        context.tradeWindow = context.i

    
    if context.canTrade:

        # If the value is over sold then it is a good time to buy
        if tsi_short[-1] <= context.TSI_OverSold and context.stakeInMarket < 1.0:
            context.stakeInMarket += .25
            order_target_percent(context.asset, context.stakeInMarket)
            print("Bought", (pos_amount*price + ((cash / 2) / price)), "amount of LTC")
            context.canTrade = False
            context.tradeWindow = context.i

        # If the market is over bought it is a good time to sell.
        if tsi_short[-1] >= context.TSI_OverBought and pos_amount >= 0.5:
            context.stakeInMarket -= .25
            order_target_percent(context.asset, context.stakeInMarket)
            print("Sold ", pos_amount, "LTC for $", (pos_amount * price))
            context.canTrade = False
            context.tradeWindow = context.i


    elif context.i >= context.tradeWindow + 1440:
            context.canTrade = True

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

    # Third chart: Compare percentage change between our portfolio
    # and the price of the asset
    ax3 = plt.subplot(513, sharex=ax1)
    perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax3)
    ax3.legend_.remove()
    ax3.set_ylabel('Percent Change')
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))


    # Fourth chart: Plot TSI
    ax4 = plt.subplot(514, sharex=ax1)
    perf.loc[:, ['tsi_long', 'tsiEMA_HighBol', 'tsiEMA_LowBol']].plot(ax=ax4, label="tsi_long")
    ax4.set_ylabel('TSI')
    #ax4.axhline(context.TSI_OverBought, color='darkgoldenrod')
    #ax4.axhline(context.TSI_OverSold, color='darkgoldenrod')
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
        base_currency='usd',
        start=pd.to_datetime('2017-04-01', utc=True),
        end=pd.to_datetime('2018-05-30', utc=True),
    )
