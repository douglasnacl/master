# run this to install all required packages to this code
# to run:
#   - go to the working directory path
#   - then use "chmod +x install_env.sh" to make this file `secure`
#   - finally run "./install_env.sh"
conda create -n stock_trading_bot_msc_douglas python=3.7 --no-default-packages # tensorflow 1.x requires python 3.7 or lower
conda activate stock_trading_bot_msc_douglas
conda install -c anaconda pip
pip install numpy pandas multipledispatch yfinance 
conda install -c anaconda tensorflow-gpu==1.15
pip install gym==2.0.0
pip install gym-anytrading==pip install gym-anytrading
pip install stable-baselines==3.7.0
pip install -U scikit-learn==1.7.3
pip install python-dotenv==0.19.2
