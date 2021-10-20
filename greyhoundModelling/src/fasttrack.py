# this wrapper is provided to make it easier to use the FastTrack API
# it is provided on an as is basis no liability is taken for any errors or issues incurred

import urllib
import xmltodict
from datetime import datetime
from datetime import timedelta
import pandas as pd
import mapping
from tqdm import tqdm
import time

def xmldict(url):
    # Reads an xml file into as a python dict
    hdr = {'User-Agent':'Mozilla/5.0'}
    req = urllib.request.Request(url, headers = hdr)
    file = urllib.request.urlopen(req)
    
    data = file.read()
    file.close()

    data = xmltodict.parse(data)
    return data

class Fasttrack():
    """
    FastTrack Data Download Centre operations
    """
    
    url = 'https://fasttrack.grv.org.au/DataExport/'
    def __init__(self, seckey):
        self.seckey = seckey
        turl = self.url + seckey + '/01_01_2021'
        try:
            testseckey = xmldict(turl)
            if testseckey.get('exception', None) == 'Invalid Security Key':
                print("Invalid Security Key")
            else:
                print("Valid Security Key")

        except Exception as exception:
            print(exception)
            print("Check you have a valid security key")

    def listTracks(self):
        """
        Returns a list of tracks, track codes and their state

        Returns
        -------
        df : DataFrame
            details of track code mapping
            
        """
        
        df = pd.DataFrame({
            'track_name': [row['trackName'] for row in mapping.trackCodes],
            'track_code': [str(row['trackCode']) for row in mapping.trackCodes],
            'state': [row['State'] for row in mapping.trackCodes]
            }).sort_values(by = ['state','track_name']).reset_index(drop = True)
        return df

    def getMeetingDetail(self, dt_start, dt_end = None, tracks = None):
        """
        Returns a DataFrame with the track codes and timeslots for each date
        specified.
        
        Parameters
        ----------
        dt_start : str
            start date for the results to retrieve in 'yyyy-mm-dd' format
        dt_end : str, optional
            end date for the results to retrieve in 'yyyy-mm-dd' format. The default is None.
        tracks : TYPE, optional
            track codes to filter on. The default is None.

        Returns
        -------
        meetingDf : DataFrame
            contains all the meets between specified dates.
            
        """
              
        if dt_end == None:
            dt_end = dt_start

        delta = datetime.strptime(dt_end, '%Y-%m-%d') - datetime.strptime(dt_start, '%Y-%m-%d')
        dt_list = [datetime.strptime(dt_start, '%Y-%m-%d') + timedelta(days = i) for i in range(delta.days + 1)]
        
        print("Getting meets for each date ..")
        time.sleep(0.5)
        meetingRows = []
        for dt in tqdm(dt_list):
            dt = datetime.strftime(dt, '%d_%m_%Y')
            try:
                meeting_dict = xmldict(self.url + self.seckey + '/' + dt)
            except Exception as exception:
                print(exception)
                print('Retrying once ..')
                time.sleep(15)
                meeting_dict = xmldict(baseUrl + seckey + '/' + dt)
            if meeting_dict.get('exception', None) == 'Invalid Date Specified':
                print("\nInvalid Date Specified .. stopping request")
                meetingDf = pd.DataFrame()
                return meetingDf
            if meeting_dict['meetings'] != None:
                if isinstance(meeting_dict['meetings']['meeting'], list):
                    for data in meeting_dict['meetings']['meeting']:
                        if isinstance(data, dict):
                            meetingRows.append(data)
                else:
                    meetingRows.append(meeting_dict['meetings']['meeting'])
        meetingDf = pd.DataFrame(meetingRows)
        if((tracks != None) & (len(meetingDf) > 0)):
            meetingDf = meetingDf[meetingDf['track'].isin(tracks)]
        return meetingDf

    def getRaceResults(self, dt_start, dt_end, tracks = None):
        """
        Retrieves all the 'Race Result Format' files from FastTrack
        and outputs them to a DataFrame
        
        Parameters
        ----------
        dt_start : str
            start date for the results to retrieve in 'yyyy-mm-dd' format
        dt_end : str
            end date for the results to retrieve in 'yyyy-mm-dd' format
        tracks : list, optional
            track codes to filter on. The default is None.

        Returns
        -------
        racesDf : DataFrame
            contains all the race details between specified dates.
        dogResultsDf : DataFrame
            contains all the dog results between specified dates.       
            
        """    
        
        meetingDf = self.getMeetingDetail(dt_start, dt_end, tracks)

        raceDetail_url = self.url + '{0}/{1}/{2}/{3}/RaceResults/XML'
        
        print("Getting historic results details ..")
        time.sleep(0.5)        
        raceRows = []
        dogRows = []
        for track, timeslot, dt in tqdm(zip(meetingDf['track'], meetingDf['timeslot'], meetingDf['date']), total = len(meetingDf['track'])):
            ft_date = datetime.strftime(datetime.strptime(dt, '%d-%b-%Y'), '%d_%m_%Y')
            try:
                raceDet = xmldict(raceDetail_url.format(self.seckey, mapping.timeslot_mapping[timeslot], ft_date, track))
            except Exception as exception:
                print(exception)
                print('Retrying once ..')
                time.sleep(15)
                raceDet = xmldict(raceDetail_url.format(self.seckey, mapping.timeslot_mapping[timeslot], ft_date, track))

            # Some meets are abandoned despite appearing in the main date call. Continue scraping
            # if the call returns a "File Not Found" exception.
            if raceDet.get('exception', None) == 'File Not Found':
                continue
            else:
                if isinstance(raceDet['Meet']['Race'], list):
                    for raceData in raceDet['Meet']['Race']:
                        raceData['Track'] = raceDet['Meet']['Track']
                        raceData['date'] = raceDet['Meet']['Date']
                        if 'Dog' in raceData:
                            for dogData in raceData['Dog']:
                                if((dogData['@id'] == "") & len(dogData) > 1):
                                    print(track + " " + dt)
                                if dogData['@id'] != "":
                                    dogData['RaceId'] = raceData['@id']
                                    dogData['TrainerId'] = dogData['Trainer'].get('@id', None)
                                    dogData['TrainerName'] = dogData['Trainer'].get('#text', None)
                                    dogData.pop('Trainer', None)
                                    dogRows.append(dogData)
                        raceData.pop('Dog', None)
                        raceData.pop('Dividends', None)
                        raceData.pop('Exotics', None)
                        raceData.pop('Times', None)
                        raceRows.append(raceData)  
                else:
                    raceData = raceDet['Meet']['Race']
                    raceData['Track'] = raceDet['Meet']['Track']
                    raceData['date'] = raceDet['Meet']['Date']
                    if 'Dog' in raceData:
                        for dogData in raceData['Dog']:
                            if((dogData['@id'] == "") & len(dogData) > 1):
                                print(track + " " + dt)
                            if dogData['@id'] != "":
                                dogData['RaceId'] = raceData['@id']
                                dogData['TrainerId'] = dogData['Trainer'].get('@id', None)
                                dogData['TrainerName'] = dogData['Trainer'].get('#text', None)
                                dogData.pop('Trainer', None)
                                dogRows.append(dogData)
                    raceData.pop('Dog', None)
                    raceData.pop('Dividends', None)
                    raceData.pop('Exotics', None)
                    raceData.pop('Times', None)
                    raceRows.append(raceData)
        racesDf = pd.DataFrame(raceRows)
        dogResultsDf = pd.DataFrame(dogRows)
        return racesDf, dogResultsDf
    
    def getBasicFormat(self, dt, tracks = None):
        """
        Retrieves the all the 'Basic Format' files for the specified date

        Parameters
        ----------
        dt : str
            date of the files you want to retrieve in 'yyyy-mm-dd' format.
        tracks : list, optional
            track codes to filter on. The default is None.

        Returns
        -------
        racesDf : DataFrame
            contains all the race details for the specified date.
        dogLineupsDf : DataFrame
            contains all the dog basic form information for the specified date.    
            
        """
        
        meetingDf = self.getMeetingDetail(dt, dt, tracks)
        
        lineup_url = self.url + '{0}/{1}/{2}/{3}/BasicPlus/XML'
        
        if len(meetingDf) == 0:
            time.sleep(0.5)
            print("No races on specified date to fetch")
            return None, None
        
        raceRows = []
        dogRows = []
        print("Getting dog lineups ..")
        time.sleep(0.5)
        for track, timeslot, dt in tqdm(zip(meetingDf['track'], meetingDf['timeslot'], meetingDf['date']), total = len(meetingDf['track'])):
            ft_date = datetime.strftime(datetime.strptime(dt, '%d-%b-%Y'), '%d_%m_%Y')
            try:
                lineupDet = xmldict(lineup_url.format(self.seckey, mapping.timeslot_mapping[timeslot], ft_date, track))
            except Exception as exception:
                print(exception)
                print('Retrying once ..')
                time.sleep(15)
                lineupDet = xmldict(lineup_url.format(self.seckey, mapping.timeslot_mapping[timeslot], ft_date, track))
            if lineupDet.get('exception', None) != 'File Not Found':
                if isinstance(lineupDet['Meet']['Race'], list):
                    for race in lineupDet['Meet']['Race']:
                        for dog in race['Dog']:
                            if(dog['BestTime'] not in ["* * * VACANT BOX * * *", "* * * NO RESERVE * * *"]):
                                dog['DamId'] = dog['Dam'].get('@id', None)
                                dog['DamName'] = dog['Dam'].get('#text', None)
                                dog['SireId'] = dog['Sire'].get('@id', None)
                                dog['SireName'] = dog['Sire'].get('#text', None)
                                dog['TrainerId'] = dog['Trainer'].get('@id', None)
                                dog['TrainerName'] = dog['Trainer'].get('#text', None)
                                dog.pop('Dam', None)
                                dog.pop('Sire', None)
                                dog.pop('Trainer', None)
                                dog['RaceId'] = race['@id']
                                dogRows.append(dog)
                        race['Track'] = lineupDet['Meet']['Track']
                        race['Date'] = lineupDet['Meet']['Date']
                        race['Quali'] = lineupDet['Meet']['Quali']
                        race['TipsComments_Bet'] = race['TipsComments']['Bet']
                        race['TipsComments_Tips'] = race['TipsComments']['Tips']
                        race.pop('Dog', None)
                        race.pop('TipsComments', None)
                        raceRows.append(race)
                else:
                    race = lineupDet['Meet']['Race']
                    for dog in race['Dog']:
                        if(dog['BestTime'] not in ["* * * VACANT BOX * * *", "* * * NO RESERVE * * *"]):
                            dog['DamId'] = dog['Dam'].get('@id', None)
                            dog['DamName'] = dog['Dam'].get('#text', None)
                            dog['SireId'] = dog['Sire'].get('@id', None)
                            dog['SireName'] = dog['Sire'].get('#text', None)
                            dog['TrainerId'] = dog['Trainer'].get('@id', None)
                            dog['TrainerName'] = dog['Trainer'].get('#text', None)
                            dog.pop('Dam', None)
                            dog.pop('Sire', None)
                            dog.pop('Trainer', None)
                            dog['RaceId'] = race['@id']
                            dogRows.append(dog)                
                    race['Track'] = lineupDet['Meet']['Track']
                    race['Date'] = lineupDet['Meet']['Date']
                    race['Quali'] = lineupDet['Meet']['Quali']
                    race['TipsComments_Bet'] = race['TipsComments']['Bet']
                    race['TipsComments_Tips'] = race['TipsComments']['Tips']
                    race.pop('Dog', None)
                    race.pop('TipsComments', None)
                    raceRows.append(race)
        racesDf = pd.DataFrame(raceRows)
        dogLineupsDf = pd.DataFrame(dogRows)
        return racesDf, dogLineupsDf
    
    def getFullFormat(self, dt, tracks = None):
        """
        Retrieves the all the 'Full Format' files for the specified date

        Parameters
        ----------
        dt : str
            date of the files you want to retrieve in 'yyyy-mm-dd' format.
        tracks : list, optional
            track codes to filter on. The default is None.

        Returns
        -------
        racesDf : DataFrame
            contains all the race details for the specified date.
        dogLineupsDf : DataFrame
            contains all the dog full form information for the specified date.  
            
        """
        
        meetingDf = self.getMeetingDetail(dt, dt, tracks)
     
        lineup_url = self.url + '{0}/{1}/{2}/{3}/FullPlus/XML'

        if len(meetingDf) == 0:
            time.sleep(0.5)
            print("No races on specified date to fetch")
            return None, None

        raceRows = []
        dogRows = []
        print("Getting dog lineups ..")
        time.sleep(0.5)
        for track, timeslot, dt in tqdm(zip(meetingDf['track'], meetingDf['timeslot'], meetingDf['date']), total = len(meetingDf['track'])):
            ft_date = datetime.strftime(datetime.strptime(dt, '%d-%b-%Y'), '%d_%m_%Y')

            try:
                lineupDet = xmldict(lineup_url.format(self.seckey, mapping.timeslot_mapping[timeslot], ft_date, track))
            except Exception as exception:
                print(exception)
                print('Retrying once ..')
                time.sleep(15)
                lineupDet = xmldict(lineup_url.format(self.seckey, mapping.timeslot_mapping[timeslot], ft_date, track))
                
            if lineupDet.get('exception', None) != 'File Not Found':
                if isinstance(lineupDet['Meet']['Race'], list):
                    for race in lineupDet['Meet']['Race']:
                        for dog in race['Dog']:
                            if(dog['BestTime'] not in ["* * * VACANT BOX * * *", "* * * NO RESERVE * * *"]):
                                dog['DamId'] = dog['Dam'].get('@id', None)
                                dog['DamName'] = dog['Dam'].get('#text', None)
                                dog['SireId'] = dog['Sire'].get('@id', None)
                                dog['SireName'] = dog['Sire'].get('#text', None)
                                dog['TrainerId'] = dog['Trainer'].get('@id', None)
                                dog['TrainerName'] = dog['Trainer'].get('#text', None)
                                dog.pop('Dam', None)
                                dog.pop('Sire', None)
                                dog.pop('Trainer', None)
                                dog['RaceId'] = race['@id']
                                dogRows.append(dog)
                        race['Track'] = lineupDet['Meet']['Track']
                        race['Date'] = lineupDet['Meet']['Date']
                        race['Quali'] = lineupDet['Meet']['Quali']
                        race['TipsComments_Bet'] = race['TipsComments']['Bet']
                        race['TipsComments_Tips'] = race['TipsComments']['Tips']
                        race.pop('Dog', None)
                        race.pop('TipsComments', None)
                        raceRows.append(race)
                else:
                    race = lineupDet['Meet']['Race']
                    for dog in race['Dog']:
                        if(dog['BestTime'] not in ["* * * VACANT BOX * * *", "* * * NO RESERVE * * *"]):
                            dog['DamId'] = dog['Dam'].get('@id', None)
                            dog['DamName'] = dog['Dam'].get('#text', None)
                            dog['SireId'] = dog['Sire'].get('@id', None)
                            dog['SireName'] = dog['Sire'].get('#text', None)
                            dog['TrainerId'] = dog['Trainer'].get('@id', None)
                            dog['TrainerName'] = dog['Trainer'].get('#text', None)
                            dog.pop('Dam', None)
                            dog.pop('Sire', None)
                            dog.pop('Trainer', None)
                            dog['RaceId'] = race['@id']
                            dogRows.append(dog)                
                    race['Track'] = lineupDet['Meet']['Track']
                    race['Date'] = lineupDet['Meet']['Date']
                    race['Quali'] = lineupDet['Meet']['Quali']
                    race['TipsComments_Bet'] = race['TipsComments']['Bet']
                    race['TipsComments_Tips'] = race['TipsComments']['Tips']
                    race.pop('Dog', None)
                    race.pop('TipsComments', None)
                    raceRows.append(race)
                    
        racesDf = pd.DataFrame(raceRows)
        dogLineupsDf = pd.DataFrame(dogRows)
        
        return racesDf, dogLineupsDf