"""
Functional backend for calling SQL functions and recording the interactions.
"""

import functools
import logging

from google.cloud import bigquery
import streamlit as st

# Create a logger that will print to stdout
logger = logging.getLogger("function_calling_app")

if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

MAXIMUM_BYTES_BILLED = 100000000  # Data limit per query job
ERROR_MSG = """
            We're having trouble running this SQL query. This
            could be due to an invalid query or the structure of
            the data. Try rephrasing your question to help the
            model generate a valid query. Details: """


def _log_function_call(func):
    """Decorator to log function calls, arguments, and return values."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        log_args = ", ".join(
            [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
        )
        name = func.__name__
        logger.info(f"Calling {name} with ({log_args})")
        try:
            result = func(self, *args, **kwargs)
            logger.info(f"{name} returned: {result}")
            self.update_message((name, log_args, result))
            return result
        except Exception as e:
            logger.error(f"Exception in {name}: {e}")
            self.update_message((name, log_args, e))
            raise  # Re-raise the exception after logging

    return wrapper


class SQLBackend:
    """Simple class that provides basic functions as well as logging interactions."""

    api_requests_and_responses: list
    backend_details: str
    dataset_id: str
    client: bigquery.Client

    def __init__(self, dataset_id: str):
        self.client = bigquery.Client()
        self.dataset_id = dataset_id

    @_log_function_call
    def list_datasets(self) -> str:
        """Get a list of datasets that will help answer the user's question."""
        api_response = self.client.list_datasets()
        api_response = self.dataset_id  # In place for demo. Remove to see all datasets
        return api_response

    @_log_function_call
    def list_tables(self, dataset_id: str) -> str:
        """List tables in a dataset that will help answer the user's question.

        Args:
            dataset_id: Dataset ID to fetch tables from.
        """
        api_response = self.client.list_tables(dataset_id)
        api_response = str([table.table_id for table in api_response])
        return api_response

    @_log_function_call
    def get_table(self, table_id: str) -> str:
        """Get information about a table, including the description, schema, and number of rows that will help answer the user's question.

        Always use the fully qualified dataset and table names.

        Args:
            table_id: Fully qualified ID of the table to get information about.
        """
        api_response = self.client.get_table(table_id)
        api_response = api_response.to_api_repr()
        api_response = [
            str(api_response.get("description", "")),
            str([column["name"] for column in api_response["schema"]["fields"]]),
        ]
        return str(api_response)

    @_log_function_call
    def sql_query(self, query: str) -> str:
        """Get information from data in BigQuery using SQL queries.

        Args:
          query: SQL query on a single line that will help give quantitative answers to the user's question when run on a BigQuery dataset and table. In the SQL query, always use the fully qualified dataset and table names.
        """
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=MAXIMUM_BYTES_BILLED)
        try:
            cleaned_query = (
                query.replace("\\n", " ").replace("\n", "").replace("\\", "")
            )
            query_job = self.client.query(cleaned_query, job_config=job_config)
            api_response = query_job.result()
            api_response = str([dict(row) for row in api_response])
            api_response = api_response.replace("\\", "").replace("\n", "")
            return api_response
        except Exception as e:
            logger.error(e.with_traceback)
            e.msg = ERROR_MSG + str(e)
            raise e

    def reset_details(self, message_placeholder):
        self.api_requests_and_responses = []
        self.message_placeholder = message_placeholder  # Updated as functions progress

    def get_backend_details(self):
        """Return saved info about function calls as string."""
        backend_details = ""
        for name, params, return_value in self.api_requests_and_responses:
            backend_details += "- Function call:\n"
            backend_details += f"   - Function name: ```{name}```"
            backend_details += "\n\n"
            backend_details += f"   - Function parameters: ```{params}```"
            backend_details += "\n\n"
            backend_details += f"   - API response: ```{return_value}```"
            backend_details += "\n\n"
        return backend_details

    def update_message(self, update_tuple):
        """Update the streamlit message_placeholder with function call history."""
        self.api_requests_and_responses.append(update_tuple)
        with self.message_placeholder.container():
            st.markdown(self.get_backend_details())
