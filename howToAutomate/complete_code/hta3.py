import pandas as pd
from flumine import BaseStrategy 
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
import logging
import betfairlightweight
from flumine import Flumine, clients
import datetime
from flumine.worker import BackgroundWorker
from flumine.events.events import TerminationEvent
import os
import csv
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes

# Will create a file called how_to_automate_3.log in our current working directory
logging.basicConfig(filename = 'how_to_automate_3.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Thoroughbred model (named the kash-ratings-model)
kash_url_1 = 'https://betfair-data-supplier-prod.herokuapp.com/api/widgets/kash-ratings-model/datasets?date='
kash_url_2 = pd.Timestamp.now().strftime("%Y-%m-%d") # todays date formatted as YYYY-mm-dd
kash_url_3 = '&presenter=RatingsPresenter&csv=true'

kash_url = kash_url_1 + kash_url_2 + kash_url_3
kash_url

# Greyhounds model (named the iggy-joey-model)
iggy_url_1 = 'https://betfair-data-supplier-prod.herokuapp.com/api/widgets/iggy-joey/datasets?date='
iggy_url_2 = pd.Timestamp.now().strftime("%Y-%m-%d")
iggy_url_3 = '&presenter=RatingsPresenter&csv=true'

iggy_url = iggy_url_1 + iggy_url_2 + iggy_url_3
iggy_url

# Download todays thoroughbred ratings
kash_df = pd.read_csv(kash_url)

## Data clearning
# Rename Columns
kash_df = kash_df.rename(columns={"meetings.races.bfExchangeMarketId":"market_id","meetings.races.runners.bfExchangeSelectionId":"selection_id","meetings.races.runners.ratedPrice":"rating"})
# Only keep columns we need
kash_df = kash_df[['market_id','selection_id','rating']]
# Convert market_id to string
kash_df['market_id'] = kash_df['market_id'].astype(str)
kash_df

# Set market_id and selection_id as index for easy referencing
kash_df = kash_df.set_index(['market_id','selection_id'])
kash_df

# e.g. can reference like this: 
    # df.loc['1.195173067'].loc['4218988']
    # to return 210.17

# Download todays greyhounds ratings
iggy_df = pd.read_csv(iggy_url)

## Data clearning
# Rename Columns
iggy_df = iggy_df.rename(columns={"meetings.races.bfExchangeMarketId":"market_id","meetings.races.runners.bfExchangeSelectionId":"selection_id","meetings.races.runners.ratedPrice":"rating"})
# Only keep columns we need
iggy_df = iggy_df[['market_id','selection_id','rating']]
# Convert market_id to string
iggy_df['market_id'] = iggy_df['market_id'].astype(str)
iggy_df

# Set market_id and selection_id as index for easy referencing
iggy_df = iggy_df.set_index(['market_id','selection_id'])
iggy_df

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

# Will create a file called how_to_automate_3.log in our current working directory
logging.basicConfig(filename = 'how_to_automate_3.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# New strategy called FlatKashModel for the Thoroughbreds model
class FlatKashModel(BaseStrategy):

    def start(self) -> None:
        print("starting strategy 'FlatKashModel'")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        if market_book.status != "CLOSED":
            return True

    # If check_market_book returns true i.e. the market is open and not closed then we will run process_market_book once initially
    # After the first inital time process_market_book has been run, every single time the market ticks, process_market_book will run again
    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        # If time is less than 1min and we haven't placed a bet yet then look at our ratings and place bet
        if market.seconds_to_start < 60 and market_book.inplay == False:
            for runner in market_book.runners:
                # Check runner hasn't scratched and that first layer of back or lay price exists
                if runner.status == "ACTIVE" and runner.ex.available_to_back[0] and runner.ex.available_to_lay[0]:                     
                    # If best available to back price is > rated price then flat $5 bet                
                    if runner.ex.available_to_back[0]['price'] > kash_df.loc[market_book.market_id].loc[runner.selection_id].item():
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
                    if runner.ex.available_to_lay[0]['price'] < kash_df.loc[market_book.market_id].loc[runner.selection_id].item():
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

# New strategy called FlatIggyModel for the Greyhound model
class FlatIggyModel(BaseStrategy):
    
    def start(self) -> None:
        print("starting strategy 'FlatIggyModel'")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        if market_book.status != "CLOSED":
            return True

    # If check_market_book returns true i.e. the market is open and not closed then we will run process_market_book once initially
    # After the first inital time process_market_book has been run, every single time the market ticks, process_market_book will run again
    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        
        # If time is less than 1min and we haven't placed a bet yet then look at our ratings and place bet
        if market.seconds_to_start < 60 and market_book.inplay == False:
            for runner in market_book.runners:
                # Check runner hasn't scratched and that first layer of back or lay price exists
                if runner.status == "ACTIVE" and runner.ex.available_to_back[0] and runner.ex.available_to_lay[0]:                
                # If best available to back price is > rated price then flat $5 bet
                    if runner.ex.available_to_back[0]['price'] > iggy_df.loc[market_book.market_id].loc[runner.selection_id].item():
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
                    if runner.ex.available_to_lay[0]['price'] < iggy_df.loc[market_book.market_id].loc[runner.selection_id].item():
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
        # logger.info("No more markets available, terminating framework")
        flumine.handler_queue.put(TerminationEvent(flumine))

logger = logging.getLogger(__name__)

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
        if os.path.exists("orders_hta_3.csv"):
            logging.info("Results file exists")
        else:
            with open("orders_hta_3.csv", "w") as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("orders_hta_3.csv", "a") as m:
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

thoroughbreds_strategy = FlatKashModel(
    market_filter=streaming_market_filter(
        event_type_ids=["7"], # Horse Racing
        country_codes=["AU"], # Australian Markets
        market_types=["WIN"], # Win Markets 
    ),
    max_order_exposure= 50, # Max bet sizes of $50
    max_trade_count=1, # Max of trade/bet attempt per selection
    max_live_trade_count=1, # Max of 1 unmatched Bet per selection
)

greyhounds_strategy = FlatIggyModel(
    market_filter=streaming_market_filter(
        event_type_ids=["4339"], # Greyhound Racing
        country_codes=["AU"], # Australian Markets
        market_types=["WIN"], # Win Markets
    ),
    max_order_exposure= 50, # Max bet sizes of $50
    max_trade_count=1, # Max of trade/bet attempt per selection
    max_live_trade_count=1, # Max of 1 unmatched Bet per selection
)

framework.add_strategy(thoroughbreds_strategy) # Add horse racing strategy to our framework
framework.add_strategy(greyhounds_strategy) # Add greyhound racing strategy to our framework

# Add the stopped to our framework
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

framework.run() # run all our strategies