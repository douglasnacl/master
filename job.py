# from utilities.io.stock_data_generator import StockDataGenerator
from utilities.io.fetch_data import fetch_data
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

parser.add_argument(
    "-v",
    "--visualize",
    help="optional: executa rotina com gráficos",
    action="store_true",
)

parser.add_argument(
    "--processing_device",
    choices=["CPU", "GPU"],
    default="GPU",
    help="optional: Escolhe o dispositivo para processamento (default: GPU)",
)

args = parser.parse_args()

if __name__ == "__main__":
    
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
        logging.info("O diretório de logs foi criado!")
    
    logging_basic_config()
    logging.info("Running the current task")

    # if args.processing_device != 'CPU':
    #     if args.processing_device != 'GPU':
    #         raise ValueError("""
    #             Apenas são permitidos valores CPU ou GPU
    #                 > O valor padrão é GPU, caso não seja passado.
    #         """)
    #     processing_device = "GPU"

    if(args.download_data):
        logging.info("Rotinha de Download de Dados")
        # stock_data = StockDataGenerator()
        # stock_data.export_csv()

        asset = 'BTCUSD'
        # Ativo a ser avaliado
        interval = '1h' # Também é possível usar '4h','1h','15m','1m'
        # Define-se a data inicial e final
        interval_start = datetime(2021, 11, 1, 0, 0)
        interval_end = datetime(2022, 11, 1, 0, 0)
        # Chama a função para obtenção dos dados
        _ = fetch_data(interval_start, interval_end, asset, interval)
   
    if(args.save_weights & args.visualize):
        logging.info("Salvando pesos")
        routine(save_weights=True, processing_device=args.processing_device, visualize=True)
    if(args.visualize):
        logging.info("Salvando pesos e Visualização")
        routine(save_weights=False, processing_device=args.processing_device, visualize=True)
    if(args.save_weights):
        logging.info("Salvando pesos e Visualização")
        routine(save_weights=True, processing_device=args.processing_device)
    else:
        routine(processing_device=args.processing_device)