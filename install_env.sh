# run this to install all required packages to this code
# to run:
#   - go to the working directory path
#   - then use "chmod +x install_env.sh" to make this file `secure`
#   - finally run "./install_env.sh"
conda create -n stock_trading_bot_msc_douglas python --no-default-packages
conda activate stock_trading_bot_msc_douglas
conda install -c python
pip install yfinance
pip install multipledispatch
pip install numpy 
pip install pandas
pip install tensorflow-gpu
pip install quandl
pip install python-dotenv
pip install matplotlib
pip install importlib-metadata
pip install gym

pip install -U scikit-learn==1.7.3
pip install python-dotenv==0.19.2
