# Import libraries for logging in
import betfairlightweight
from flumine import Flumine, clients
from flumine import BaseStrategy 
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
import pandas as pd
import numpy as np
import logging
import datetime
from flumine.worker import BackgroundWorker
from flumine.events.events import TerminationEvent
import os
import csv
import logging
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes

logging.basicConfig(filename = 'how_to_automate_2.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(lineno)d:%(message)s')

# Import libraries for logging in
import betfairlightweight
from flumine import Flumine, clients

# Credentials to login and logging in 
trading = betfairlightweight.APIClient('username','password',app_key='appkey')
client = clients.BetfairClient(trading, interactive_login=True)

# Login
framework = Flumine(client=client)

# Code to login when using security certificates
# trading = betfairlightweight.APIClient('username','password',app_key='appkey', certs=r'C:\Users\zhoui\openssl_certs')
# client = clients.BetfairClient(trading)

# framework = Flumine(client=client)

class BackFavStrategy(BaseStrategy):

    # Defines what happens when we start our strategy i.e. this method will run once when we first start running our strategy
    def start(self) -> None:
        print("starting strategy 'BackFavStrategy'")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        if market_book.status != "CLOSED":
            return True

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        
        # Collect data on last price traded and the number of bets we have placed
        snapshot_last_price_traded = []
        snapshot_runner_context = []
        for runner in market_book.runners:
                snapshot_last_price_traded.append([runner.selection_id,runner.last_price_traded])
                # Get runner context for each runner
                runner_context = self.get_runner_context(
                    market.market_id, runner.selection_id, runner.handicap
                )
                snapshot_runner_context.append([runner_context.selection_id, runner_context.executable_orders, runner_context.live_trade_count, runner_context.trade_count])

        # Convert last price traded data to dataframe
        snapshot_last_price_traded = pd.DataFrame(snapshot_last_price_traded, columns=['selection_id','last_traded_price'])
        # Find the selection_id of the favourite
        snapshot_last_price_traded = snapshot_last_price_traded.sort_values(by = ['last_traded_price'])
        fav_selection_id = snapshot_last_price_traded['selection_id'].iloc[0]
        logging.info(snapshot_last_price_traded) # logging

        # Convert data on number of bets we have placed to a dataframe
        snapshot_runner_context = pd.DataFrame(snapshot_runner_context, columns=['selection_id','executable_orders','live_trade_count','trade_count'])
        logging.info(snapshot_runner_context) # logging

        for runner in market_book.runners:
            if runner.status == "ACTIVE" and market.seconds_to_start < 60 and market_book.inplay == False and runner.selection_id == fav_selection_id and snapshot_runner_context.iloc[:,1:].sum().sum() == 0:
                trade = Trade(
                    market_id=market_book.market_id,
                    selection_id=runner.selection_id,
                    handicap=runner.handicap,
                    strategy=self,
                )
                order = trade.create_order(
                    side="BACK", order_type=LimitOrder(price=runner.last_price_traded, size=5)
                )
                market.place_order(order)

logger = logging.getLogger(__name__)

"""
Worker can be used as followed:
    framework.add_worker(
        BackgroundWorker(
            framework,
            terminate,
            func_kwargs={"today_only": True, "seconds_closed": 1200},
            interval=60,
            start_delay=60,
        )
    )
This will run every 60s and will terminate 
the framework if all markets starting 'today' 
have been closed for at least 1200s
"""

# Function that stops automation running at the end of the day
def terminate(
    context: dict, flumine, today_only: bool = True, seconds_closed: int = 600
) -> None:
    """terminate framework if no markets
    live today.
    """
    markets = list(flumine.markets.markets.values())
    markets_today = [
        m
        for m in markets
        if m.market_start_datetime.date() == datetime.datetime.utcnow().date()
        and (
            m.elapsed_seconds_closed is None
            or (m.elapsed_seconds_closed and m.elapsed_seconds_closed < seconds_closed)
        )
    ]
    if today_only:
        market_count = len(markets_today)
    else:
        market_count = len(markets)
    if market_count == 0:
        logger.info("No more markets available, terminating framework")
        flumine.handler_queue.put(TerminationEvent(flumine))

FIELDNAMES = [
    "bet_id",
    "strategy_name",
    "market_id",
    "selection_id",
    "trade_id",
    "date_time_placed",
    "price",
    "price_matched",
    "size",
    "size_matched",
    "profit",
    "side",
    "elapsed_seconds_executable",
    "order_status",
    "market_note",
    "trade_notes",
    "order_notes",
]

class LiveLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, *args, **kwargs):
        super(LiveLoggingControl, self).__init__(*args, **kwargs)
        self._setup()

    # Changed file path and checks if the file orders_hta_2.csv already exists, if it doens't then create it
    def _setup(self):
        if os.path.exists("orders_hta_2.csv"):
            logging.info("Results file exists")
        else:
            with open("orders_hta_2.csv", "w") as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("orders_hta_2.csv", "a") as m:
            for order in orders:
                if order.order_type.ORDER_TYPE == OrderTypes.LIMIT:
                    size = order.order_type.size
                else:
                    size = order.order_type.liability
                if order.order_type.ORDER_TYPE == OrderTypes.MARKET_ON_CLOSE:
                    price = None
                else:
                    price = order.order_type.price
                try:
                    order_data = {
                        "bet_id": order.bet_id,
                        "strategy_name": order.trade.strategy,
                        "market_id": order.market_id,
                        "selection_id": order.selection_id,
                        "trade_id": order.trade.id,
                        "date_time_placed": order.responses.date_time_placed,
                        "price": price,
                        "price_matched": order.average_price_matched,
                        "size": size,
                        "size_matched": order.size_matched,
                        "profit": 0 if not order.cleared_order else order.cleared_order.profit,
                        "side": order.side,
                        "elapsed_seconds_executable": order.elapsed_seconds_executable,
                        "order_status": order.status.value,
                        "market_note": order.trade.market_notes,
                        "trade_notes": order.trade.notes_str,
                        "order_notes": order.notes_str,
                    }
                    csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                    csv_writer.writerow(order_data)
                except Exception as e:
                    logger.error(
                        "_process_cleared_orders_meta: %s" % e,
                        extra={"order": order, "error": e},
                    )

        logger.info("Orders updated", extra={"order_count": len(orders)})

    def _process_cleared_markets(self, event):
        cleared_markets = event.event
        for cleared_market in cleared_markets.orders:
            logger.info(
                "Cleared market",
                extra={
                    "market_id": cleared_market.market_id,
                    "bet_count": cleared_market.bet_count,
                    "profit": cleared_market.profit,
                    "commission": cleared_market.commission,
                },
            )

strategy = BackFavStrategy(
    market_filter=streaming_market_filter(
        event_type_ids=["4339"], # Greyhounds
        country_codes=["AU"], # Australian Markets
        market_types=["WIN"], # Win Markets
    ),
    max_trade_count=1, # max total number of trades per runner
    max_live_trade_count=1, # max live (with executable orders) trades per runner
    max_selection_exposure=20, # max exposure of 20 per horse
    max_order_exposure= 20 # Max bet sizes of $20
)

framework.add_strategy(strategy)

# Add the auto terminate to our framework
framework.add_worker(
    BackgroundWorker(
        framework,
        terminate,
        func_kwargs={"today_only": True, "seconds_closed": 1200},
        interval=60,
        start_delay=60,
    )
)

framework.add_logging_control(
    LiveLoggingControl()
)

framework.run() # run our framework