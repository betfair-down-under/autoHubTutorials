# Import libraries
import glob
import os
import time
import logging
import csv
import math
from pythonjsonlogger import jsonlogger
from concurrent import futures
from flumine import FlumineSimulation, clients, utils
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes
from strategies.back_fav import BackFavStrategy

# Logging
logger = logging.getLogger()
custom_format = "%(asctime) %(levelname) %(message)"
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(custom_format)
formatter.converter = time.gmtime
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)  # Set to logging.CRITICAL to speed up simulation

# Import necessary libraries
from flumine import BaseStrategy 
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook

import pandas as pd
import numpy as np
import logging

# Create a new strategy as a new class called BackFavStrategy, this in turn will allow us to create a new Python object later
    # BackFavStrategy is a child class inhereting from a predefined class in Flumine we imported above called BaseStrategy
class BackFavStrategy(BaseStrategy):
    # Defines what happens when we start our strategy i.e. this method will run once when we first start running our strategy
    def start(self) -> None:
        # We will want to change what is printed with we have multiple strategies
        print("starting strategy 'BackFavStrategy'")

    # Defines what happens when we first look at a market
    # This method will prevent looking at markets that are closed
    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        if market_book.status != "CLOSED":
            return True

    # If check_market_book returns true i.e. the market is open and not closed then we will run process_market_book once initially
    # After the first inital time process_market_book has been run, every single time the market ticks, process_market_book will run again
    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        
        # Find last traded price as a dataframe
        snapshot_last_price_traded = []
        snapshot_runner_context = []
        for runner in market_book.runners:
                snapshot_last_price_traded.append([runner.selection_id,runner.last_price_traded])
                # Get runner context for each runner
                runner_context = self.get_runner_context(
                    market.market_id, runner.selection_id, runner.handicap
                )
                snapshot_runner_context.append([runner_context.selection_id, runner_context.executable_orders, runner_context.live_trade_count, runner_context.trade_count])

        snapshot_last_price_traded = pd.DataFrame(snapshot_last_price_traded, columns=['selection_id','last_traded_price'])
        snapshot_last_price_traded = snapshot_last_price_traded.sort_values(by = ['last_traded_price'])
        fav_selection_id = snapshot_last_price_traded['selection_id'].iloc[0]

        snapshot_runner_context = pd.DataFrame(snapshot_runner_context, columns=['selection_id','executable_orders','live_trade_count','trade_count'])

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

# Log results from simulation into csv file named sim_hta_2.csv
# If the csv file doesn't exist then it is created, otherwise we append results to the csv file
class BacktestLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, *args, **kwargs):
        super(BacktestLoggingControl, self).__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        if os.path.exists("sim_hta_2.csv"):
            logging.info("Results file exists")
        else:
            with open("sim_hta_2.csv", "w") as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("sim_hta_2.csv", "a") as m:
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
    strategy = BackFavStrategy(
        # market_filter selects what portion of the historic data we simulate our strategy on
        # markets selects the list of betfair historic data files
        # market_types specifies the type of markets
        # listener_kwargs specifies the time period we simulate for each market
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