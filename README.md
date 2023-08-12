# master

To train the agent, we need to set up a simple game with a limited set of options, a relatively low-dimensional state, and other parameters that can be easily modified and extended.

More specifically, the environment samples a stock price time series for a single ticker using a random start date to simulate a trading period that, by default, contains 252 days, or 1 year. The state contains the (scaled) price and volume, as well as some technical indicators like the percentile ranks of price and volume, a relative strength index (RSI), as well as 5- and 21-day returns. The agent can choose from three actions:

- **Buy**: Invest capital for a long position in the stock
- **Flat**: Hold cash only
- **Sell short**: Take a short position equal to the amount of capital

The environment accounts for trading cost, which is set to 10bps by default. It also deducts a 1bps time cost per period. It tracks the net asset value (NAV) of the agent's portfolio and compares it against the market portfolio (which trades frictionless to raise the bar for the agent).


# Command-line arguments

The properties that you can use while is parsing command-line arguments and options are: 

- `--download_data:` This is an optional argument that takes in a string value. The value must be either "BTCUSD" or "ETHUSD". If this argument is provided, the program will execute a routine to obtain new data. If this argument is not provided, the default value "BTCUSD" will be used.
- `--visualize:` This is an optional argument that does not take in any value. If this argument is provided, the program will execute a routine with graphs.
- `--processing_device:` This is an optional argument that takes in a string value. The value must be either "CPU" or "GPU". If this argument is provided, the program will choose the device for processing. If this argument is not provided, the default value


To Do

- Verificar frequencia de treinamento da rede
    - epsilon_decay
- Verificar ações e suas consequencias
- Verificar redes similares em outros lugares e a forma como aprendem
- Verificar parametros que podem ser ajustados
- Verificar a taxa de aprendizado