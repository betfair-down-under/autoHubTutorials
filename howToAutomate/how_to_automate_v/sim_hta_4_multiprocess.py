# Import libraries
import glob
import os
import time
import logging
import csv
import pandas as pd
import json
import math
from pythonjsonlogger import jsonlogger
from flumine import FlumineSimulation, BaseStrategy, utils, clients
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus
from flumine.order.ordertype import OrderTypes
from flumine.markets.market import Market
from flumine.controls.loggingcontrols import LoggingControl
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
from pythonjsonlogger import jsonlogger
from concurrent import futures

# Logging
logger = logging.getLogger()
custom_format = "%(asctime) %(levelname) %(message)"
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(custom_format)
formatter.converter = time.gmtime
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)  # Set to logging.CRITICAL to speed up simulation

# Read in predictions from hta_4
todays_data = pd.read_csv('backtest.csv', dtype = ({"market_id":str}))
todays_data = todays_data.set_index(['market_id','selection_id'])

### New implementation
class FlatBetting(BaseStrategy):
    def start(self) -> None:
        print("starting strategy 'FlatBetting' using the model we created the Greyhound modelling in Python Tutorial")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        if market_book.status != "CLOSED":
            return True

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:

        # At the 60 second mark:
        if market.seconds_to_start < 60 and market_book.inplay == False:

            # Can't simulate polling API
            # Only use streaming API:
            for runner in market_book.runners:
                model_price = todays_data.loc[market.market_id].loc[runner.selection_id]['rating']
                # If best available to back price is > rated price then flat $5 back
                if runner.status == "ACTIVE" and runner.ex.available_to_back[0]['price'] > model_price:
                    trade = Trade(
                    market_id=market_book.market_id,
                    selection_id=runner.selection_id,
                    handicap=runner.handicap,
                    strategy=self,
                    )
                    order = trade.create_order(
                        side="BACK", order_type=LimitOrder(price=runner.ex.available_to_back[0]['price'], size=5.00)
                    )
                    market.place_order(order)
                # If best available to lay price is < rated price then flat $5 lay
                if runner.status == "ACTIVE" and runner.ex.available_to_lay[0]['price'] < model_price:
                    trade = Trade(
                    market_id=market_book.market_id,
                    selection_id=runner.selection_id,
                    handicap=runner.handicap,
                    strategy=self,
                    )
                    order = trade.create_order(
                        side="LAY", order_type=LimitOrder(price=runner.ex.available_to_lay[0]['price'], size=5.00)
                    )
                    market.place_order(order)

# Fields we want to log in our simulations
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

# Log results from simulation into csv file named sim_hta_4.csv
# If the csv file doesn't exist then it is created, otherwise we append results to the csv file
class BacktestLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, *args, **kwargs):
        super(BacktestLoggingControl, self).__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        if os.path.exists("sim_hta_4.csv"):
            logging.info("Results file exists")
        else:
            with open("sim_hta_4.csv", "w") as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("sim_hta_4.csv", "a") as m:
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
                        "profit": order.simulated.profit,
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

# Searches for all betfair data files within the folder sample_monthly_data_output
data_folder = 'sample_monthly_data_output'
data_files = os.listdir(data_folder,)
data_files = [f'{data_folder}/{path}' for path in data_files]

def run_process(markets):
    """Replays a Betfair historic data. Places bets according to the user defined strategy and tries to accurately simulate matching by replaying the historic data.

    Args:
        markets (list: [file paths]): a list of file paths to where the historic data is stored locally. e.g. user/zhoui/downloads/test.csv
    """    
    # Set Flumine to simulation mode
    client = clients.SimulatedClient()
    framework = FlumineSimulation(client=client)

    # Set parameters for our strategy
    strategy = FlatBetting(
        market_filter={
            "markets": markets,  
            'market_types':['WIN'],
            "listener_kwargs": {"inplay": False, "seconds_to_start": 80},  
            },
        max_order_exposure=1000,
        max_selection_exposure=1000,
        max_live_trade_count=1,
        max_trade_count=1,
    )
    # Run our strategy on the simulated market
    framework.add_strategy(strategy)
    framework.add_logging_control(
        BacktestLoggingControl()
    )
    framework.run()

# Multi processing
if __name__ == "__main__":
    all_markets = data_files  # All the markets we want to simulate
    processes = os.cpu_count()  # Returns the number of CPUs in the system.
    markets_per_process = 8   # 8 is optimal as it prevents data leakage.

    _process_jobs = []
    with futures.ProcessPoolExecutor(max_workers=processes) as p:
        # Number of chunks to split the process into depends on the number of markets we want to process and number of CPUs we have.
        chunk = min(
            markets_per_process, math.ceil(len(all_markets) / processes)
        )
        # Split all the markets we want to process into chunks to run on separate CPUs and then run them on the separate CPUs
        for m in (utils.chunks(all_markets, chunk)):
            _process_jobs.append(
                p.submit(
                    run_process,
                    markets=m,
                )
            )
        for job in futures.as_completed(_process_jobs):
            job.result()  # wait for result