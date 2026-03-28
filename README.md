# Mapping Electrochemical Hydrogen Production Research with NLP: Automated Relevance Filtering, Transformer-Based Topic Modeling, and Scientometric Trends

Este repositório contém os códigos e informações suplementares para o artigo científico **"[Insira o Título do seu Artigo Aqui]"**. O estudo utiliza técnicas de Processamento de Linguagem Natural (NLP) para analisar o panorama da pesquisa global sobre produção de hidrogênio.

## ⚠️ Aviso sobre a Disponibilidade dos Dados (Data Availability)

Os dados originais utilizados nesta pesquisa foram obtidos através da base de dados **Web of Science (Clarivate)**. Devido a restrições de licenciamento e aos *Terms of Business* da Clarivate, os dados brutos (metadados completos, resumos, etc.) não podem ser redistribuídos publicamente neste repositório.

No entanto, para garantir a total **transparência e reprodutibilidade** do nosso estudo, fornecemos uma lista contendo todos os **DOIs (Digital Object Identifiers)** dos documentos analisados. 

Pesquisadores com acesso institucional à Web of Science podem utilizar a lista de DOIs fornecida para buscar e baixar o dataset original por conta própria, permitindo a execução dos códigos aqui disponibilizados.

## 📂 Estrutura do Repositório

* `data/`: Contém a lista de DOIs dos artigos incluídos no estudo (ex: `lista_dois.csv`).
* `src/`: Pasta contendo os códigos-fonte desenvolvidos para o estudo.
  * `01_preprocessing.py` (ou .ipynb): Códigos utilizados para limpar e pré-processar os dados brutos.
  * `02_nlp_analysis.py`: Códigos para a modelagem e análise textual.
  * `03_generate_reports.py`: Códigos para gerar os gráficos, tabelas e relatórios finais apresentados no artigo.
* `requirements.txt`: Lista de bibliotecas e dependências necessárias para rodar o código em Python.

## 🚀 Como reproduzir a pesquisa

1. **Obtenção dos Dados:** * Baixe o arquivo com a lista de DOIs localizada na pasta `data/`. 
   * Acesse a Web of Science e utilize a ferramenta de "Busca Avançada" para buscar pelos registros usando os identificadores (exemplo de query: `DO=(doi1 OR doi2 OR ...)`). 
   * Exporte os resultados no formato utilizado pelos scripts e salve na pasta `data/raw/` (que você deverá criar localmente).
2. **Configuração do Ambiente:** * Clone este repositório para a sua máquina local.
   * Instale as dependências executando:
     ```bash
     pip install -r requirements.txt
     ```
3. **Execução:** * Rode os scripts sequencialmente localizados na pasta `src/` para replicar o pré-processamento, a análise de NLP e a geração de relatórios.

## 📄 Licença e Uso

Os códigos e scripts de autoria própria desenvolvidos para este projeto estão licenciados sob a [MIT License](LICENSE) (ou insira a licença que preferir, como Apache 2.0). Sinta-se à vontade para utilizá-los e adaptá-los, desde que o artigo original seja devidamente citado.

**Citação do artigo:**
> Azevedo, L. C., et al. (Ano). *Título do Artigo*. Nome da Revista. DOI: [Link do DOI do seu paper, quando publicado]
