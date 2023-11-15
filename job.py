from utilities.io.fetch_data import fetch_data
from utilities.methods import routine
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import logging
import os

logs_path = "logs/"

def logging_basic_config():
    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"logs/{datetime.now().date()}-stock-trading-bot.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.CRITICAL,
        datefmt="%d-%b-%y %H:%M:%S",
    )

parser = argparse.ArgumentParser()
parser.add_argument(
    "-dd",
    "--download_data",
    choices=["BTCUSD", "ETHUSD"],
    help="optional: executa rotina para obter novos dados (default: BTCUSD)",
)

parser.add_argument(
    "-i",
    "--interval",
    choices=["deterministic", "stochastic"],
    help="optional: Escolhe o tipo de período (default: stochastic)",
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
    np.random.seed(42)
    tf.random.set_seed(42)

    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
        logging.info("O diretório de logs foi criado!")
    
    logging_basic_config()
    logging.info("Running the current task")

    # Definindo o periodo
    if args.interval != 'deterministic':
        if args.interval == 'stochastic':
            args.interval = "stochastic"
        else:
            args.interval = "deterministic"
        print("Periodo: ", args.interval, "\n\n")
        
    # Definindo o dispositivo de processamento
    if args.processing_device != 'CPU':
        if args.processing_device != 'GPU':
            raise ValueError("""
                Apenas são permitidos valores CPU ou GPU
                    > O valor padrão é GPU, caso não seja passado.
            """)
        args.processing_device = "GPU"
    
    _deterministic = True if args.interval == 'deterministic' else False
    if(args.visualize):
        logging.info("Visualização")
        routine(processing_device=args.processing_device, visualize=True, deterministic=_deterministic)

    elif(args.download_data):
        logging.info("Rotina de Download de Dados")

        asset = args.download_data
        # Ativo a ser avaliado
        interval = '1h' # Também é possível usar '4h','1h','15m','1m'
        # Define-se a data inicial e final
        interval_start = datetime(2016, 3, 14, 0, 0)
        interval_end = datetime(2023, 9, 9, 0, 0)
        # Chama a função para obtenção dos dados
        _ = fetch_data(interval_start, interval_end, asset, interval)
    else:
        routine(processing_device=args.processing_device, deterministic=_deterministic)

    