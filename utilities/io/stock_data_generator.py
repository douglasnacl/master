from multipledispatch import dispatch
import yfinance as yf
from datetime import datetime
import logging
import os
import pandas as pd

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
   
        try:
            self.database = self.connect()
        except Exception as e:
            raise e
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
            logging.info(f"Getting the historical data - period {self.PERIOD} and interval {self.INTERVAL} for the last {self.PERIOD} days")
            dataframe =  pd.DataFrame(self.database.history(period=self.PERIOD, interval=self.INTERVAL))
            return dataframe
        except Exception as e:
            raise e

    @dispatch(datetime, datetime)
    def get_data(self, start_date: datetime, end_date: datetime):
        try: 
            logging.info(f"Getting the historical data - period{self.PERIOD} and interval {self.INTERVAL} \n from {start_date} to {end_date}")
            dataframe = self.database.history(period=self.PERIOD, interval=self.INTERVAL, start=start_date, end=end_date)
            return dataframe
        except Exception as e:
            raise e
    
    def export_csv(self):
        
        path = ASSETS_URL
        file_name = OUTPUT_FILE_NAME
        pathfile = ASSETS_URL + OUTPUT_FILE_NAME + '-'+ str(datetime.now().date()) + '.csv'
        
        dataframe = self.get_data()
        if not os.path.exists(path):
            logging.info(f"The folder to store the file has being created")
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