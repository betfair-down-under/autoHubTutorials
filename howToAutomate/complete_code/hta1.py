# Import libraries for logging in
import betfairlightweight
from flumine import Flumine, clients
from flumine import BaseStrategy 
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
import logging 

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

logging.basicConfig(filename = 'how_to_automate_1.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Create a new strategy as a new class called LayStrategy, this in turn will allow us to create a new Python object later
    # LayStrategy is a child class inheriting from a class in Flumine we imported above called BaseStrategy
class LayStrategy(BaseStrategy):
    # Defines what happens when we start our strategy i.e. this method will run once when we first start running our strategy
    def start(self) -> None:
        print("starting strategy 'LayStrategy'")

    # Prevent looking at markets that are closed
    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        if market_book.status != "CLOSED":
            return True

    # If check_market_book returns true i.e. the market is open and not closed then we will run process_market_book once initially
    #  After the first initial time, process_market_book runs every single time someone places, updates or cancels a bet
    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        for runner in market_book.runners: # Loops through each of the runners in the race
            if runner.status == "ACTIVE": # If the runner is active (hasen't been scratched)
                # Place a lay bet at a price of 1.01 with $5 volume
                trade = Trade(
                    market_id=market_book.market_id, # The market_id for the specific market
                    selection_id=runner.selection_id, # The selection_id of the horse/dog/team
                    handicap=runner.handicap, # The handicap of the horse/dog/team
                    strategy=self, # Strategy this bet is part of: itself (LayStrategy)
                )
                order = trade.create_order(
                    side="LAY", order_type=LimitOrder(price=1.01, size=5.00) # Lay bet, Limit order price of 1.01 size = $5
                )
                market.place_order(order) # Place the order

strategy = LayStrategy(
    market_filter=streaming_market_filter(
        event_type_ids=["4339"], # Greyhounds
        country_codes=["AU"], # Australia
        market_types=["WIN"], # Win Markets
    )
)

framework.add_strategy(strategy)

framework.run()