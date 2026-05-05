from dotenv import load_dotenv
import os
import urllib.parse
import clickhouse_connect

class ClickHouseClient:
    def __init__(self):
        load_dotenv()
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
        password = urllib.parse.quote_plus(os.environ.get("clk_prod_pass"))

        self.client = clickhouse_connect.get_client(
            host=host,
            username=username,
            password=password,
            port=443
        )
    def query(self, q):
        return self.client.query_df(q)
