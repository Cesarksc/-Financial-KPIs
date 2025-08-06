import os
import logging
import locale
from datetime import datetime

from dotenv import load_dotenv
import requests
import pandas as pd
from rapidfuzz import process, fuzz

def log_error_decorator(func):
    """
    Decorator para registrar erros usando o método log_error da instância.

    Args:
        func (Callable): Função a ser decorada.

    Returns:
        Callable: Função decorada.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logging.error(str(e))
            raise
    return wrapper

def measure_time(func):
    """
    Decorator para medir o tempo de execução de uma função.

    Args:
        func (Callable): Função a ser decorada.

    Returns:
        Callable: Função decorada que mede o tempo de execução.
    """
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        elapsed = end - start
        logging.info(f"Tempo de execução: {elapsed}")
        print(f"Tempo de execução: {elapsed}")
        return result
    return wrapper


class CSVDataFrameLoader:
    """
    Classe para carregar DataFrames a partir de arquivos CSV.

    Args:
        csv_path (dict[str, str]): Dicionário com caminhos dos arquivos CSV.
    """
    def __init__(self, csv_path: dict[str, str]):
        self.csv_path: dict[str, str] = csv_path

    def load_dataframe(self, file_type: str) -> pd.DataFrame:
        """
        Carrega um DataFrame a partir de um arquivo CSV.

        Args:
            file_type (str): Tipo do arquivo ('revenue' ou 'expenses').

        Returns:
            pd.DataFrame: DataFrame carregado.

        Raises:
            ValueError: Se o tipo de arquivo for inválido.
            FileNotFoundError: Se o arquivo não existir.
            pd.errors.ParserError: Se houver erro de parsing.
            UnicodeDecodeError: Se houver erro de encoding.
        """
        if file_type not in self.csv_path:
            logging.error(f"[ERRO] Tipo de arquivo inválido solicitado: {file_type}")
            raise ValueError(f"Tipo de arquivo inválido: {file_type}")

        file_path = self.csv_path[file_type]
        logging.info(f"[INFO] Iniciando leitura do arquivo CSV de {file_type}: {file_path}")
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
            logging.info(f"[INFO] Arquivo CSV de {file_type} carregado com sucesso. Linhas: {len(df)}")
            return df
        except FileNotFoundError:
            logging.error(f"[ERRO] Arquivo de {file_type} não encontrado: {file_path}")
            raise
        except pd.errors.ParserError as e:
            logging.error(f"[ERRO] Erro de parsing no arquivo de {file_type}: {e}")
            raise
        except UnicodeDecodeError as e:
            logging.error(f"[ERRO] Erro de encoding no arquivo de {file_type}: {e}")
            raise
        except Exception as e:
            logging.error(f"[ERRO] Erro inesperado ao ler {file_type}: {e}")
            raise


class DataFrameValidator:
    """
    Classe utilitária para validar e concatenar DataFrames de receitas e despesas.

    Args:
        df_revenue (pd.DataFrame): DataFrame de receitas.
        df_expenses (pd.DataFrame): DataFrame de despesas.
    """
    def __init__(self, df_revenue: pd.DataFrame, df_expenses: pd.DataFrame):
        self.df_revenue = df_revenue
        self.df_expenses = df_expenses

    def same_number_of_columns(self) -> bool:
        """
        Verifica se os DataFrames possuem o mesmo número de colunas.

        Returns:
            bool: True se possuem o mesmo número de colunas, False caso contrário.
        """
        return self.df_revenue.shape[1] == self.df_expenses.shape[1]

    def same_column_names(self) -> bool:
        """
        Verifica se os DataFrames possuem os mesmos nomes de colunas (ordem e nomes).

        Returns:
            bool: True se os nomes das colunas são iguais, False caso contrário.
        """
        return list(self.df_revenue.columns) == list(self.df_expenses.columns)

    def concatenate(self) -> pd.DataFrame:
        """
        Concatena os DataFrames se forem compatíveis.

        Returns:
            pd.DataFrame: DataFrame único com os dados empilhados.

        Raises:
            ValueError: Se os DataFrames não forem compatíveis.
        """
        if not self.same_number_of_columns():
            logging.error("[ERRO] Os DataFrames não possuem o mesmo número de colunas.")
            raise ValueError("Os DataFrames não possuem o mesmo número de colunas.")
        if not self.same_column_names():
            logging.error("[ERRO] Os DataFrames não possuem os mesmos nomes de colunas.")
            raise ValueError("Os DataFrames não possuem os mesmos nomes de colunas.")
        logging.info("[INFO] Concatenando DataFrames de receitas e despesas.")
        return pd.concat([self.df_revenue, self.df_expenses], ignore_index=True)


class DataFrameCleaner:
    def __init__(self, df_base: pd.DataFrame, valid_restaurants: list[str]):
        """
        Inicializa o limpador de DataFrame.

        Args:
            df_base (pd.DataFrame): DataFrame a ser limpo.
            valid_restaurants (list[str]): Lista de restaurantes válidos.
        """
        self.df_base = df_base
        self.valid_restaurants = valid_restaurants

    def validate_and_format_date_column(self, date_column: str = 'data') -> None:
        """
        Valida e padroniza a coluna de data para o formato dd/mm/aaaa.

        Args:
            date_column (str, optional): Nome da coluna de data a ser validada e formatada. Default é 'data'.

        Raises:
            ValueError: Se houver datas inválidas, informando o número de linhas afetadas.
        """
        if date_column not in self.df_base.columns:
            logging.error(f"[ERRO] Coluna '{date_column}' não encontrada no DataFrame.")
            raise ValueError(f"Coluna '{date_column}' não encontrada no DataFrame.")

        def try_parse_date(val):
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(str(val), fmt)
                except Exception:
                    continue
            return pd.NaT

        self.df_base[date_column] = self.df_base[date_column].apply(try_parse_date)

        invalid_dates = self.df_base[self.df_base[date_column].isna()]
        if not invalid_dates.empty:
            num_invalid = invalid_dates.shape[0]
            logging.error(f"[ERRO] Foram encontradas {num_invalid} linhas com data inválida na coluna '{date_column}'.")
            raise ValueError(f"Foram encontradas {num_invalid} linhas com data inválida na coluna '{date_column}'.")

        self.df_base[date_column] = self.df_base[date_column].dt.strftime("%d/%m/%Y")

    def validate_restaurant_column(self, restaurant_column: str = 'restaurante', similarity_threshold: int = 70) -> None:
        """
        Valida e corrige automaticamente os valores da coluna de restaurante usando fuzzy matching.

        Args:
            restaurant_column (str, optional): Nome da coluna de restaurante. Default é 'restaurante'.
            similarity_threshold (int, optional): Percentual mínimo de similaridade para corrigir automaticamente. Default é 70.
        """
        if restaurant_column not in self.df_base.columns:
            logging.error(f"[ERRO] Coluna '{restaurant_column}' não encontrada no DataFrame.")
            raise ValueError(f"Coluna '{restaurant_column}' não encontrada no DataFrame.")

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

        self.df_base[restaurant_column] = self.df_base[restaurant_column].apply(get_best_match)

    def validate_tipo_column(self, tipo_column: str = 'tipo', similarity_threshold: int = 70) -> None:
        """
        Valida e corrige automaticamente os valores da coluna 'tipo' usando fuzzy matching para 'RECEITA' ou 'DESPESA'.

        Args:
            tipo_column (str, optional): Nome da coluna a ser validada. Default é 'tipo'.
            similarity_threshold (int, optional): Percentual mínimo de similaridade para corrigir automaticamente. Default é 70.
        """
        if tipo_column not in self.df_base.columns:
            logging.error(f"[ERRO] Coluna '{tipo_column}' não encontrada no DataFrame.")
            raise ValueError(f"Coluna '{tipo_column}' não encontrada no DataFrame.")

        valores_validos = ["RECEITA", "DESPESA"]

        def get_best_match(tipo):
            tipo = str(tipo).strip().upper()
            if tipo in valores_validos:
                return tipo

            match, score, _ = process.extractOne(
                tipo,
                valores_validos,
                scorer=fuzz.ratio
            )
            if score >= similarity_threshold:
                return match
            return tipo

        self.df_base[tipo_column] = self.df_base[tipo_column].apply(get_best_match)

    def validate_and_format_columns(self, date_column: str = 'data', restaurant_column: str = 'restaurante', tipo_column: str = 'tipo', similarity_threshold: int = 70) -> None:
        """
        Valida e formata as colunas de data, restaurante e tipo.

        Args:
            date_column (str, optional): Nome da coluna de data. Default é 'data'.
            restaurant_column (str, optional): Nome da coluna de restaurante. Default é 'restaurante'.
            tipo_column (str, optional): Nome da coluna de tipo. Default é 'tipo'.
            similarity_threshold (int, optional): Percentual mínimo de similaridade para correção automática. Default é 70.

        Returns:
            pd.DataFrame: DataFrame validado e formatado.
        """
        self.validate_and_format_date_column(date_column)
        self.validate_restaurant_column(restaurant_column, similarity_threshold)
        self.validate_tipo_column(tipo_column, similarity_threshold)

        return self.df_base


class CNPJAPI:
    def __init__(self, api_url: str):
        """
        Classe utilitária para buscar nome social de CNPJs.

        Args:
            api_url (str): URL base da API de consulta de CNPJ.
        """
        self.api_url = api_url

    def get_social_name(self, cnpj: str) -> str:
        """
        Consulta a API e retorna o nome social do restaurante.

        Args:
            cnpj (str): CNPJ do restaurante.

        Returns:
            str: Nome social do restaurante ou vazio se não encontrado.
        """
        try:
            logging.info(f"[INFO] Consultando API para CNPJ: {cnpj}")
            response = requests.get(self.api_url + cnpj, timeout=5)
            response.raise_for_status()
            data = response.json()
            nome = data.get('razao_social', '')
            if nome:
                logging.info(f"[INFO] Nome social encontrado para CNPJ {cnpj}: {nome}")
            else:
                logging.warning(f"[WARN] Nome social não encontrado para CNPJ {cnpj}.")
            return nome
        except Exception as e:
            logging.error(f"[ERRO] Erro ao consultar API para CNPJ {cnpj}: {e}")
            return ''

    def add_social_name_column(self, df_base: pd.DataFrame, cnpj_column: str = 'cnpj_fornecedor', new_column: str = 'nome_social') -> pd.DataFrame:
        """
        Adiciona uma coluna com o nome social baseado no CNPJ.

        Args:
            df_base (pd.DataFrame): DataFrame de entrada.
            cnpj_column (str, optional): Nome da coluna do CNPJ. Default é 'cnpj_fornecedor'.
            new_column (str, optional): Nome da nova coluna a ser criada. Default é 'nome_social'.

        Returns:
            pd.DataFrame: DataFrame com a nova coluna adicionada.
        """
        if cnpj_column not in df_base.columns:
            logging.error(f"[ERRO] Coluna '{cnpj_column}' não encontrada no DataFrame.")
            raise ValueError(f"Coluna '{cnpj_column}' não encontrada no DataFrame.")
        
        cnpj_to_social = {}
        def get_name(cnpj):
            cnpj = str(cnpj).zfill(14)
            if cnpj in cnpj_to_social:
                return cnpj_to_social[cnpj]
            name = self.get_social_name(cnpj)
            cnpj_to_social[cnpj] = name
            return name
        logging.info(f"[INFO] Adicionando coluna '{new_column}' com nome social baseado no CNPJ.")
        df_base[new_column] = df_base[cnpj_column].apply(get_name)
        return df_base
    

class KPIAnalyzer:
    def __init__(self, df_base):
        self.df_base = df_base

    def receita_total(self):
        return self.df_base.loc[self.df_base['tipo'] == 'RECEITA', 'valor'].sum()

    def despesa_total(self):
        return self.df_base.loc[self.df_base['tipo'] == 'DESPESA', 'valor'].sum()

    def margem_operacional(self):
        receita = self.receita_total()
        despesa = self.despesa_total()
        return (receita - despesa) / receita * 100 if receita > 0 else 0

    def receita_por_restaurante(self):
        return self.df_base[self.df_base['tipo'] == 'RECEITA'].groupby('restaurante')['valor'].sum()

    def despesa_por_restaurante(self):
        return self.df_base[self.df_base['tipo'] == 'DESPESA'].groupby('restaurante')['valor'].sum()

    def ticket_medio(self):
        receitas = self.df_base[self.df_base['tipo'] == 'RECEITA']
        return receitas['valor'].sum() / len(receitas) if not receitas.empty else 0

    def percentual_marketing(self):
        marketing = self.df_base[
            (self.df_base['tipo'] == 'DESPESA') &
            (self.df_base['subcategoria'].str.contains("Marketing", case=False, na=False))
        ]
        return (marketing['valor'].sum() / self.receita_total()) * 100 if self.receita_total() > 0 else 0


    def format_currency(self, value):
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def __str__(self):
        dados = [
            ["Receita Total", self.format_currency(self.receita_total())],
            ["Despesa Total", self.format_currency(self.despesa_total())],
            ["Margem Operacional (%)", f"{self.margem_operacional():.2f}%".replace(".", ",")],
            ["Ticket Médio", self.format_currency(self.ticket_medio())],
            ["% Marketing sobre Receita", f"{self.percentual_marketing():.2f}%".replace(".", ",")]
        ]
        df_indicadores = pd.DataFrame(dados, columns=["Indicador", "Valor"])
        return df_indicadores.to_string(index=False)



class Main:
    """
    Classe principal do script.

    Args:
        csv_paths (dict[str, str]): Dicionário com caminhos dos arquivos CSV.
        valid_restaurants (list[str]): Lista de restaurantes válidos.
        api_url (str): URL base da API de consulta de CNPJ.
    """
    def __init__(self, csv_paths: dict[str, str], valid_restaurants: list[str], api_url: str):
        self.df_revenue: pd.DataFrame = None
        self.df_expenses: pd.DataFrame = None
        self.api_url = api_url
        self.valid_restaurants = valid_restaurants

        self.csv_data_loader_instance = CSVDataFrameLoader(csv_paths)

    @log_error_decorator

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executa o processamento principal do script.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: DataFrames de receitas e despesas.
        """
        logging.info("[INFO] Iniciando carregamento dos DataFrames de receitas e despesas.")
        self.df_revenue = self.csv_data_loader_instance.load_dataframe('revenue')
        self.df_expenses = self.csv_data_loader_instance.load_dataframe('expenses')

        logging.info("[INFO] Validando e concatenando DataFrames.")
        self.dataframe_validator_instance = DataFrameValidator(self.df_revenue, self.df_expenses)
        self.df_base = self.dataframe_validator_instance.concatenate()

        logging.info("[INFO] Limpando e validando colunas do DataFrame base.")
        self.dataframe_cleaner_instance = DataFrameCleaner(self.df_base, self.valid_restaurants)
        self.df_base = self.dataframe_cleaner_instance.validate_and_format_columns()

        logging.info("[INFO] Adicionando nome social dos CNPJs.")
        self.cnpjapi = CNPJAPI(self.api_url)
        self.df_base = self.cnpjapi.add_social_name_column(self.df_base)

        logging.info("[INFO] Calculando KPIs.")
        self.kpi_analyzer_instance = KPIAnalyzer(self.df_base)
        print(self.kpi_analyzer_instance)

        return self.df_revenue, self.df_expenses


class GlobalConfigs:
    """
    Configurações globais do script.

    Args:
        locale_setting (str, optional): Locale do sistema. Default é 'pt_BR.UTF-8'.
    """

    def __init__(self, locale_setting: str = 'pt_BR.UTF-8'):
        """
        Inicializa configurações globais.

        Args:
            locale_setting (str, optional): Locale do sistema. Default é 'pt_BR.UTF-8'.
        """
        log_dir: str = os.path.join(os.path.dirname(__file__), "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_path: str = os.path.join(log_dir, 'logging_dre_report.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            locale.setlocale(locale.LC_ALL, locale_setting)
        except locale.Error:
            logging.error(f"Não foi possível definir o locale para {locale_setting}.")

        env_path: str = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(dotenv_path=env_path, override=True)

    def get_csv_paths(self) -> dict[str, str]:
        """
        Retorna os caminhos dos arquivos de despesas e receitas definidos no .env.

        Returns:
            dict[str, str]: Dicionário com os caminhos dos arquivos CSV.
        """
        return {
            "expenses": os.getenv("CSV_DESPESAS"),
            "revenue": os.getenv("CSV_RECEITAS")
        }

    def return_valid_restaurants(self) -> list[str]:
        """
        Retorna a lista de restaurantes válidos.

        Returns:
            list[str]: Lista de restaurantes válidos.
        """
        return [
            "Sorveteria Tropical",
            "Churrascaria Boi Feliz",
            "Pizzaria Massa Fina",
            "Restaurante Veggie Life"
        ]

    def log_error(self, error_message: str) -> None:
        """
        Registra erro no log.

        Args:
            error_message (str): Mensagem de erro.
        """
        logging.error(error_message)
        print(f"Erro registrado: {error_message}")


@measure_time
def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Função principal do script.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames de receitas e despesas.
    """
    global_configs = GlobalConfigs()
    csv_paths = global_configs.get_csv_paths()
    valid_restaurants = global_configs.return_valid_restaurants()
    api_url = f"https://publica.cnpj.ws/cnpj/"
    main_instance = Main(csv_paths, valid_restaurants, api_url)
    df_revenue, df_expenses = main_instance.run()
    return df_revenue, df_expenses

if __name__ == "__main__":
    main()