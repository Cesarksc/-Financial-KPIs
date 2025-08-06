# DRE Report Analyzer

Este projeto é um script Python para análise de receitas e despesas de restaurantes, com foco em boas práticas de programação, organização de código e automação de validações de dados. O objetivo é demonstrar habilidades de desenvolvimento, utilizando técnicas modernas e código limpo.

## Principais Funcionalidades
- **Carregamento de dados**: Lê arquivos CSV de receitas e despesas definidos em variáveis de ambiente.
- **Validação e limpeza**: Valida datas, nomes de restaurantes e tipos de transação usando fuzzy matching.
- **Enriquecimento de dados**: Consulta uma API pública para obter o nome social de CNPJs.
- **Cálculo de KPIs**: Calcula indicadores financeiros como receita total, despesa total, margem operacional, ticket médio e percentual de marketing.
- **Logging estruturado**: Todos os passos e erros são registrados em arquivo de log.
- **Organização em classes**: O código é modular, facilitando manutenção e expansão.

## Estrutura do Projeto
- `main.py`: Script principal com todas as classes e funções.
- `.env`: Arquivo de variáveis de ambiente (não incluso, exemplo abaixo).
- `log/`: Pasta onde são salvos os logs de execução.

## Exemplos de Código e Boas Práticas

### 1. Organização em Classes
O código é dividido em classes para separar responsabilidades:
```python
class CSVDataFrameLoader:
    """
    Classe para carregar DataFrames a partir de arquivos CSV.
    """
    def load_dataframe(self, file_type: str) -> pd.DataFrame:
        # ...
```
**Por quê?**
Facilita a manutenção, testes e reutilização do código.

### 2. Uso de Decorators para Logging e Medição de Tempo
```python
def log_error_decorator(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logging.error(str(e))
            raise
    return wrapper
```
**Por quê?**
Automatiza o registro de erros, tornando o código mais limpo e robusto.

### 3. Validação Inteligente com Fuzzy Matching
```python
from rapidfuzz import process, fuzz
# ...
def validate_restaurant_column(self, restaurant_column: str = 'restaurante', similarity_threshold: int = 70):
    def get_best_match(restaurante):
        restaurante = str(restaurante).strip().upper()
        if restaurante in self.valid_restaurants:
            return restaurante
        match, score, _ = process.extractOne(
            restaurante,
            self.valid_restaurants,
            scorer=fuzz.ratio
        ) if self.valid_restaurants else (None, 0, None)
        if score >= similarity_threshold:
            return match
        return restaurante
```
**Por quê?**
Corrige automaticamente nomes digitados incorretamente, melhorando a qualidade dos dados.

### 4. Consulta a API Externa
```python
class CNPJAPI:
    def get_social_name(self, cnpj: str) -> str:
        response = requests.get(self.api_url + cnpj, timeout=5)
        # ...
```
**Por quê?**
Enriquece os dados com informações externas, agregando valor à análise.

### 5. Cálculo de KPIs e Apresentação
```python
class KPIAnalyzer:
    def __str__(self):
        dados = [
            ["Receita Total", self.format_currency(self.receita_total())],
            # ...
        ]
        df_indicadores = pd.DataFrame(dados, columns=["Indicador", "Valor"])
        return df_indicadores.to_string(index=False)
```
**Por quê?**
Apresenta os resultados de forma clara e formatada, facilitando a interpretação.

**Apresentação dos KPIS**
                Indicador           Valor
            Receita Total R$ 2.386.079,42
            Despesa Total R$ 1.958.414,02
   Margem Operacional (%)          17,92%
             Ticket Médio       R$ 989,66
% Marketing sobre Receita          21,45%

**Observação**
Atualmente, os indicadores são apresentados diretamente no console. No entanto, o programa foi modularizado de forma que seja simples incluir esses indicadores em dashboards, exportá-los para outros formatos de arquivo, salvar em tabelas de banco de dados ou até mesmo integrá-los a ferramentas como Power BI, Looker, entre outras.

## Como Executar
1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
2. Atualize o arquivo `.env` com os caminhos dos arquivos CSV:
   ```env
   CSV_DESPESAS=caminho/para/despesas.csv
   CSV_RECEITAS=caminho/para/receitas.csv
   ```
3. Execute o script:
   ```
   python main.py
   ```

## Observações
- O script gera logs detalhados em `log/logging_dre_report.txt`.
- O locale padrão é `pt_BR.UTF-8`, mas pode ser ajustado conforme o sistema.
- O código está pronto para expansão, com fácil inclusão de novos KPIs ou validações.
- As docstrings e registros de logging foram inicialmente gerados por IA e posteriormente revisados por humanos.