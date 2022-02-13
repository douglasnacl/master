from multipledispatch import dispatch
import yfinance as yf
from datetime import datetime
import logging
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

STOCK_NAME = os.getenv("STOCK_NAME")
OUTPUT_FILE_NAME = os.getenv("OUTPUT_FILE_NAME")
PERIOD = os.getenv("PERIOD")
INTERVAL = os.getenv("INTERVAL")
ASSETS_URL = 'assets/'

class StockDataGenerator(yf.Ticker):
    
    def __init__(self):
        self.STOCK_NAME = STOCK_NAME
        self.period = PERIOD
        self.interval = INTERVAL
        
    def connect(self):
        try:
            return yf.Ticker(self.STOCK_NAME)
        except OverflowError as e:
            raise e
    
    @property
    def period(self, period: str):
        return self.PERIOD
    @property
    def interval(self, interval: str):
        return self.INTERVAL

    @period.setter
    def period(self, period: str):
        self.PERIOD = period

    @interval.setter    
    def interval(self, interval: str):
        self.INTERVAL = interval
        
    
    @dispatch()
    def get_data(self) -> pd.DataFrame:
        try:
            database = self.connect()
            dataframe =  pd.DataFrame(database.history(period=self.PERIOD, interval=self.INTERVAL))
            dataframe = dataframe.reset_index()
            try:
                return dataframe[['Date', 'Open','High','Low','Close']]
            except:
                return dataframe[['Datetime',  'Open','High','Low','Close']]
        except:
            pass

    @dispatch(datetime, datetime)
    def get_data(self, start_date: datetime, end_date: datetime):
        try: 
            database = self.connect()
            dataframe = self.database.history(period=self.PERIOD, interval=self.INTERVAL, start=start_date, end=end_date)
            dataframe = dataframe.reset_index()
            try:
                return dataframe[['Date', 'Open','High','Low','Close']]
            except:
                return dataframe[['Datetime',  'Open','High','Low','Close']]
        except :
            pass
    
    def export_csv(self):
        
        path = ASSETS_URL
        file_name = OUTPUT_FILE_NAME
        pathfile = ASSETS_URL + OUTPUT_FILE_NAME
        
        dataframe = self.get_data()
        if not os.path.exists(path):
            os.mkdir(path)
        files_present = os.path.isfile(pathfile) 
        if not files_present:
            logging.info(f"The file {file_name} is being written")
            dataframe.to_csv(pathfile)
            print(dataframe)
        else:
            logging.info("The file already exists, it will be deleted and rewritten")
            os.remove(pathfile)
            dataframe.to_csv(pathfile)
