# Import libraries
import glob
import os
import time
import logging
import csv
from pythonjsonlogger import jsonlogger
from flumine import FlumineSimulation, clients
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
        "markets": data_files,  
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