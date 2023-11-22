# Dissertação de Mestrado: Desenvolvimento de Modelo de Inteligência Artificial para Day Trading com Criptoativos baseado em Double Deep Q-Learning

Este repositório contém o código fonte associado à minha dissertação de mestrado intitulada "Desenvolvimento de Modelo de Inteligência Artificial para Day Trading com Criptoativos baseado em Double Deep Q-Learning". Meu nome é Douglas de Oliveira, e esta pesquisa foi realizada sob a orientação do Prof. Weslley Luiz da Silva Assis, D.Sc./MCCT-UFF, e co-orientação da Profª Vanessa Silva Garcia, D.Sc./MCCT-UFF.

# Resumo

O objetivo principal desta dissertação foi desenvolver um modelo de inteligência artificial aplicado a estratégias de day trading utilizando criptoativos. A abordagem escolhida é baseada em Double Deep Q-Learning, uma técnica avançada de aprendizado por reforço profundo. O foco específico recai sobre a aplicação prática desta abordagem em um ambiente dinâmico e volátil como o mercado de criptoativos.

## Estrutura do Repositório

- **assets**: Dados relacionados ao trading.
  - **ts**: Dados temporais.
    - `BTCUSD_1h_full.csv`
    - `ETHUSD_1h_full.csv`

- **env**: Configurações de ambiente.

- **logs**: Arquivos de log e são da forma
  - `yyyy-MM-dd-stock-trading-bot.log`

- **runs**: Resultados de execuções do modelo.
  - `yyyy-MM-dd_ddqn_trader`
    - Modelos treinados, logs e parâmetros.
  - `csv`
    - Resultados em formato CSV.

  - `data`: Dados gerados à partir do csv

  - `images`: Imagens relacionadas aos resultados.

- **test**: Scripts de teste.

- **utilities**: Utilitários relacionados ao projeto.
  - **environment**: Códigos relacionados ao ambiente de trading.
  - **io**: Códigos relacionados à entrada/saída (obtenção dos dados).
  - `methods.py`: Métodos utilitários.
  - **nn**: Códigos relacionados a geração das redes neurais.
  - **rl**: Códigos relacionados à aprendizagem por reforço (agente de aprendizado por reforço).
  - **utils**: Utilitários diversos.
    - Vale uma menção especial ao `build-table.ipynb` utilizado para gerar os gráficos que utilizamos na apresentação do trabalho

- **job.py**: Código necessário a execução do modelo. `python job.py --help` para mais informações.

- **LICENSE**: Licença do projeto.

- **README.md**: Documentação principal do repositório.

- **requirements.txt**: Lista de dependências necessárias para o projeto.

- **test.txt**: Arquivo de teste.

- **training_2022-07-19-77d58568-9e56-4099-9a37-f7c66dfa70a8-1658284483.csv**: Possível arquivo de dados relacionado ao treinamento do modelo.


# Pré-requisitos

As dependencias necessárias podem ser instaladas com uso do script `/utilities/utils/install_env.sh` 
 - `observações`: este script instala as dependencias necessárias para o projeto
    - é necessário ter o python3 instalado
    - é necessário ter o pip3 instalado
    - é necessário ter o virtualenv instalado
    - é necessário ter o git instalado
 - `disclaimer`: este script foi testado em um ambiente linux (ubuntu 22.04 e manjaro) e pode não funcionar em outros ambientes ou mesmo necessitar de ajustes para funcionar em outros ambientes


# Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues, propor melhorias ou enviar pull requests.

# Agradecimentos
Gostaria de expressar minha gratidão aos meus orientadores, Prof. Weslley Luiz da Silva Assis e Profª Vanessa Silva Garcia, por sua orientação valiosa e apoio ao longo deste trabalho.

# Contato
Nome: Douglas de Oliveira
E-mail: douglas_oliveira@id.uff.br
Não hesite em entrar em contato para qualquer dúvida, sugestão ou discussão relacionada a este projeto.