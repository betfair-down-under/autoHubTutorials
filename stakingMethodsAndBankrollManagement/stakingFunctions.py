import random
import numpy as np

def FixedStakesSim(stakeSize, bankroll, ruin, minBet, maxBet, edge, bankObj):
    """
    Fixed Stakes Simulation
    
    Parameters
    ----------
    stakeSize : float
        stake size for each bet
    bankroll : float
        starting bankroll
    ruin : float
        amount where if bankroll drops below, you are ruined
    minBet : float
        the minimum bet odds which you will bet at
    maxBet : float
        the maximum bet odds which you will bet at
    edge : float
        assumed edge for each bet
    bankObj : float
        your bank objective expressed as a multiple of your starting bank
    """
    
    betRange = maxBet - minBet
    dynamicBank = bankroll
    
    # Simulate bets until either objective is achieved or ruined, cap at 50,000 bets
    for i in range(50000):
        betOdds = round(random.uniform(0,1) * betRange + minBet,2)
        winChance = (1 + edge)/betOdds
        rand = random.uniform(0,1)
        outcome = "Exhausted Bets"
        if rand < winChance:
            betPnl = stakeSize * (betOdds - 1)
        else:
            betPnl = -stakeSize
        dynamicBank = dynamicBank + betPnl
        if dynamicBank > bankObj * bankroll:
            outcome = "Objective Achieved"
            break
        if dynamicBank < ruin:
            outcome = "Ruined"
            break
        if i == 49999:
            outcome  = "Bets exhausted"
    return [outcome, i]

simStore = []

# Simulate 10,000 times
for i in range(10000):
    simStore.append(FixedStakesSim(20, 1000, 5, 1.5, 5, 0.05, 4))

probSuccess = len([i[0] for i in simStore if i[0] == 'Objective Achieved']) / len(simStore)
probRuined = len([i[0] for i in simStore if i[0] == 'Ruined']) / len(simStore)
numbetsSuccess = np.median([i[1] for i in simStore if i[0] == 'Objective Achieved'])
numbetsRuined = np.median([i[1] for i in simStore if i[0] == 'Ruined'])


def ProportionalStakesSimA(stakepct, bankroll, ruin, minBet, maxBet, edge, bankObj):
    """
    Proportional stakes simulation (staking a % of bankroll)
    
    Parameters
    ----------
    stakepct : float
        the % of your dynamic bankroll that you're staking
    bankroll : float
        starting bankroll
    ruin : float
        amount where if bankroll drops below, you are ruined
    minBet : float
        the minimum bet odds which you will bet at
    maxBet : float
        the maximum bet odds which you will bet at
    edge : float
        assumed edge for each bet
    bankObj : float
        your bank objective expressed as a multiple of your starting bank
    """
    
    betRange = maxBet - minBet
    dynamicBank = bankroll
    for i in range(50000):
        stakeSize = max(stakepct * dynamicBank, ruin)
        betOdds = round(random.uniform(0,1) * betRange + minBet,2)
        winChance = (1 + edge)/betOdds
        rand = random.uniform(0,1)
        if rand < winChance:
            betPnl = stakeSize * (betOdds - 1)
        else:
            betPnl = -stakeSize
        dynamicBank = dynamicBank + betPnl
        if dynamicBank > bankObj * bankroll:
            outcome = "Objective Achieved"
            break
        if dynamicBank < ruin:
            outcome = "Ruined"
            break
        if i == 49999:
            outcome  = "Bets exhausted"
    return [outcome, i]

simStore = []

# Simulate 10,000 times
for i in range(10000):
    simStore.append(ProportionalStakesSimA(0.02, 1000, 5, 1.5, 5, 0.05, 4))

probSuccess = len([i[0] for i in simStore if i[0] == 'Objective Achieved']) / len(simStore)
probRuined = len([i[0] for i in simStore if i[0] == 'Ruined']) / len(simStore)
numbetsSuccess = np.median([i[1] for i in simStore if i[0] == 'Objective Achieved'])
numbetsRuined = np.median([i[1] for i in simStore if i[0] == 'Ruined'])


def ProportionalStakesSimB(winpct, bankroll, ruin, minBet, maxBet, edge, bankObj):
    """
    Proportional stakes simulation (staking to win a certain % of bankroll)
    
    Parameters
    ----------
    winpct : float
        the % of your dynamic bankroll that you're staking to win
    bankroll : float
        starting bankroll
    ruin : float
        amount where if bankroll drops below, you are ruined
    minBet : float
        the minimum bet odds which you will bet at
    maxBet : float
        the maximum bet odds which you will bet at
    edge : float
        assumed edge for each bet
    bankObj : float
        your bank objective expressed as a multiple of your starting bank
    """
    
    betRange = maxBet - minBet
    dynamicBank = bankroll
    for i in range(50000):
        betOdds = round(random.uniform(0,1) * betRange + minBet,2)
        stakeSize = max((dynamicBank * winpct) / (betOdds - 1), ruin)
        winChance = (1 + edge)/betOdds
        rand = random.uniform(0,1)
        if rand < winChance:
            betPnl = stakeSize * (betOdds - 1)
        else:
            betPnl = -stakeSize
        dynamicBank = dynamicBank + betPnl
        if dynamicBank > bankObj * bankroll:
            outcome = "Objective Achieved"
            break
        if dynamicBank < ruin:
            outcome = "Ruined"
            break
        if i == 49999:
            outcome  = "Bets exhausted"
    return [outcome, i]

simStore = []

# Simulate 10,000 times
for i in range(10000):
    simStore.append(ProportionalStakesSimB(0.05, 1000, 5, 1.3, 5, 0.05, 4))

probSuccess = len([i[0] for i in simStore if i[0] == 'Objective Achieved']) / len(simStore)
probRuined = len([i[0] for i in simStore if i[0] == 'Ruined']) / len(simStore)
numbetsSuccess = np.median([i[1] for i in simStore if i[0] == 'Objective Achieved'])
numbetsRuined = np.median([i[1] for i in simStore if i[0] == 'Ruined'])



def Martingale(winamt, bankroll, ruin, minBet, maxBet, edge, bankObj):
    """
    Martingale staking simulation
    
    Parameters
    ----------
    winamt : float
        the desired win amount for each betting run
    bankroll : float
        starting bankroll
    ruin : float
        amount where if bankroll drops below, you are ruined
    minBet : float
        the minimum bet odds which you will bet at
    maxBet : float
        the maximum bet odds which you will bet at
    edge : float
        assumed edge for each bet
    bankObj : float
        your bank objective expressed as a multiple of your starting bank
    """
    
    betRange = maxBet - minBet
    dynamicBank = bankroll
    martingale_win = 1
    martingale_progressive_loss = 0
    for i in range(50000):
        betOdds = round(random.uniform(0,1) * betRange + minBet,2)
        stakeSize = max((winamt - martingale_progressive_loss) / (betOdds - 1), ruin)
        winChance = (1 + edge)/betOdds
        rand = random.uniform(0,1)
        outcome = "Exhausted Bets"
        if rand < winChance:
            betPnl = stakeSize * (betOdds - 1)
            martingale_win = 1
            martingale_progressive_loss = 0
        else:
            betPnl = -stakeSize
            martingale_win = 0
            martingale_progressive_loss =  martingale_progressive_loss - stakeSize
        dynamicBank = dynamicBank + betPnl
        if dynamicBank > bankObj * bankroll:
            outcome = "Objective Achieved"
            break
        if dynamicBank < ruin:
            outcome = "Ruined"
            break
    return [outcome, i]

simStore = []

# Simulate 10,000 times
for i in range(10000):
    simStore.append(Martingale(20, 1000, 5, 1.5, 5, 0.05, 4))

probSuccess = len([i[0] for i in simStore if i[0] == 'Objective Achieved']) / len(simStore)
probRuined = len([i[0] for i in simStore if i[0] == 'Ruined']) / len(simStore)
numbetsSuccess = np.median([i[1] for i in simStore if i[0] == 'Objective Achieved'])
numbetsRuined = np.median([i[1] for i in simStore if i[0] == 'Ruined'])


def KellyStake(bankroll, ruin, minBet, maxBet, edge, bankObj, partialKelly):
    """
    Kelly staking simulation
    
    Parameters
    ----------

    bankroll : float
        starting bankroll
    ruin : float
        amount where if bankroll drops below, you are ruined
    minBet : float
        the minimum bet odds which you will bet at
    maxBet : float
        the maximum bet odds which you will bet at
    edge : float
        assumed edge for each bet
    bankObj : float
        your bank objective expressed as a multiple of your starting bank
    partialKelly: float
        proportion of kelly staking to bet
    """
    
    betRange = maxBet - minBet
    dynamicBank = bankroll
    for i in range(50000):
        betOdds = round(random.uniform(0,1) * betRange + minBet,2)
        winChance = (1 + edge)/betOdds
        stakeSize = max(((((betOdds - 1) * winChance) - (1-winChance)) / (betOdds - 1)) * dynamicBank * partialKelly,ruin) 
        rand = random.uniform(0,1)
        outcome = "Exhausted Bets"
        if rand < winChance:
            betPnl = stakeSize * (betOdds - 1)
        else:
            betPnl = -stakeSize
        dynamicBank = dynamicBank + betPnl
        if dynamicBank > bankObj * bankroll:
            outcome = "Objective Achieved"
            break
        if dynamicBank < ruin:
            outcome = "Ruined"
            break
    return [outcome, i]

simStore = []

# Simulate 10,000 times
for i in range(10000):
    simStore.append(KellyStake(1000, 5, 1.5, 5 ,0.05, 4, 1))

probSuccess = len([i[0] for i in simStore if i[0] == 'Objective Achieved']) / len(simStore)
probRuined = len([i[0] for i in simStore if i[0] == 'Ruined']) / len(simStore)
numbetsSuccess = np.median([i[1] for i in simStore if i[0] == 'Objective Achieved'])
numbetsRuined = np.median([i[1] for i in simStore if i[0] == 'Ruined'])
