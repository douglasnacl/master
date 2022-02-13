from sympy import arg
from utilities.io.stock_data_generator import StockDataGenerator
import logging
import os
from datetime import datetime
import argparse


logs_path = "logs/"

def logging_basic_config():
    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"logs/{datetime.now().date()}-stock-trading-bot.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        datefmt="%d-%b-%y %H:%M:%S",
    )

parser = argparse.ArgumentParser()
parser.add_argument(
    "-dd",
    "--download_data",
    help="OPCIONAL : executa rotina para obter novos dados",
)

args = vars(parser.parse_args())
if __name__ == "__main__":
    
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
        logging.info("O diretório de logs foi criado!")
    
    logging_basic_config()
    logging.info("Running the current task")

    if(args.download_data):
        stock_data = StockDataGenerator()
        stock_data.export_csv()