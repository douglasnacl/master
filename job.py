from utilities.io.stock_data_generator import StockDataGenerator
from utilities.methods import routine
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
    help="optional: executa rotina para obter novos dados",
    action="store_true",
)
parser.add_argument(
    "-sw",
    "--save_weights",
    help="optional: executa rotina para salvar pesos",
    action="store_true",
)

args = parser.parse_args()

if __name__ == "__main__":
    
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
        logging.info("O diretório de logs foi criado!")
    
    logging_basic_config()
    logging.info("Running the current task")

    if(args.download_data):
        stock_data = StockDataGenerator()
        stock_data.export_csv()
    if(args.save_weights):
        routine(save_weights=True)
    else:
        routine()