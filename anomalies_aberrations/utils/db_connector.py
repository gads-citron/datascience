from dotenv import load_dotenv
import os
import urllib.parse
import clickhouse_connect

class ClickHouseClient:
    def __init__(self, env):
        if env =='prod':
            self.get_prod()
        else:
            self.get_stage()

    def get_prod(self):
        load_dotenv()
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        username = urllib.parse.quote_plus(os.environ.get("usr_name"))
        password = urllib.parse.quote_plus(os.environ.get("clk_prod_pass"))
        self.connect(host, username, password)

    def get_stage(self):
        load_dotenv()
        host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
        username = urllib.parse.quote_plus(os.environ.get("usr_name"))
        password = urllib.parse.quote_plus(os.environ.get("clk_stg_pass"))
        self.connect(host, username, password)

    def connect(self, host, username, password):
        self.client = clickhouse_connect.get_client(
            host=host,
            username=username,
            password=password,
            port=443
        )

    def query(self, q):
        return self.client.query_df(q)

    def command(self, q):
        return self.client.command(q)
