# Import libraries
import glob
import os
import time
import logging
import csv
import pandas as pd
from pythonjsonlogger import jsonlogger
from flumine import FlumineSimulation, BaseStrategy, utils, clients
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus
from flumine.order.ordertype import OrderTypes
from flumine.markets.market import Market
from flumine.controls.loggingcontrols import LoggingControl
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook

# Logging
logger = logging.getLogger()
custom_format = "%(asctime) %(levelname) %(message)"
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(custom_format)
formatter.converter = time.gmtime
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)  # Set to logging.CRITICAL to speed up simulation

def download_iggy_ratings(date):
    """Downloads the Betfair Iggy model ratings for a given date and formats it into a nice DataFrame.

    Args:
        date (datetime): the date we want to download the ratings for
    """
    iggy_url_1 = 'https://betfair-data-supplier-prod.herokuapp.com/api/widgets/iggy-joey/datasets?date='
    iggy_url_2 = date.strftime("%Y-%m-%d")
    iggy_url_3 = '&presenter=RatingsPresenter&csv=true'
    iggy_url = iggy_url_1 + iggy_url_2 + iggy_url_3

    # Download todays greyhounds ratings
    iggy_df = pd.read_csv(iggy_url)

    # Data clearning
    iggy_df = iggy_df.rename(columns={"meetings.races.bfExchangeMarketId":"market_id","meetings.races.runners.bfExchangeSelectionId":"selection_id","meetings.races.runners.ratedPrice":"rating"})
    iggy_df = iggy_df[['market_id','selection_id','rating']]
    iggy_df['market_id'] = iggy_df['market_id'].astype(str)

    # Set market_id and selection_id as index for easy referencing
    iggy_df = iggy_df.set_index(['market_id','selection_id'])
    return(iggy_df)

# Download historical ratings over a time period and convert into a big DataFrame.
back_test_period = pd.date_range(start='2022/02/27', end='2022/03/05')
frames = [download_iggy_ratings(day) for day in back_test_period]
iggy_df = pd.concat(frames)
print(iggy_df)

# Create strategy, this is the exact same strategy shown in How to Automate III
class FlatIggyModel(BaseStrategy):
    def start(self) -> None:
        print("starting strategy 'FlatIggyModel'")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        if market_book.status != "CLOSED":
            return True

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        if market.seconds_to_start < 60 and market_book.inplay == False:
            for runner in market_book.runners:
                if runner.status == "ACTIVE" and runner.ex.available_to_back[0]['price'] > iggy_df.loc[market_book.market_id].loc[runner.selection_id].item():
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
                if runner.status == "ACTIVE" and runner.ex.available_to_lay[0]['price'] < iggy_df.loc[market_book.market_id].loc[runner.selection_id].item():
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

# Log results from simulation into csv file named sim_hta_3.csv
# If the csv file doesn't exist then it is created, otherwise we append results to the csv file
class BacktestLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, *args, **kwargs):
        super(BacktestLoggingControl, self).__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        if os.path.exists("sim_hta_3.csv"):
            logging.info("Results file exists")
        else:
            with open("sim_hta_3.csv", "w") as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("sim_hta_3.csv", "a") as m:
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
strategy = FlatIggyModel(
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