import logging
from typing import Optional, Dict, Any, Tuple

# You need to install openai: pip install openai
try:
    import openai
except ImportError:
    openai = None

class NLToSQLConverter:
    """
    Converts natural language queries to optimized SQL using LLMs (e.g., GPT-4).
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            openai_api_key: OpenAI API key (required if using OpenAI's GPT models)
            model: Model to use (default: 'gpt-4')
            logger: Logger instance
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        if openai_api_key:
            if openai is not None:
                openai.api_key = openai_api_key
            else:
                raise ImportError("openai package is not installed.")
        self.available = openai is not None and openai_api_key is not None

    def translate_nl_to_sql(
        self,
        question: str,
        table_schema: str,
        temperature: float = 0,
        max_tokens: int = 512,
        user_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Translate a natural language question into an optimized SQL query.

        Args:
            question: The user's question in plain English.
            table_schema: The SQL schema (CREATE TABLE statement or DDL).
            temperature: Sampling temperature for LLM.
            max_tokens: Maximum tokens for the response.
            user_id: Optional user identifier for traceability.

        Returns:
            Tuple of (sql_query, clarifying_question if ambiguity detected else None)
        """
        if not self.available:
            raise RuntimeError("OpenAI API key is not set or openai package not available.")

        prompt = self._build_prompt(question, table_schema)

        self.logger.info(f"User[{user_id}] NL question: {question}")
        self.logger.info(f"Schema provided:\n{table_schema}")
        self.logger.debug(f"LLM prompt:\n{prompt}")

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a world-class SQL expert and language model. Given a natural language question and a SQL table schema, output a single, safe, efficient SQL query. If the question is ambiguous, ask a clarifying sub-question instead of producing SQL. NEVER generate queries that could be used for SQL injection. Only use columns present in the schema."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=None,
            )
            llm_output = response.choices[0].message["content"]
            self.logger.info(f"LLM response: {llm_output}")

            # Check for ambiguity or clarifying question
            if "clarify" in llm_output.lower() or "could you specify" in llm_output.lower() or "ambiguous" in llm_output.lower() or llm_output.strip().endswith("?"):
                clarifying = self._extract_clarifying_question(llm_output)
                sql_query = ""
            else:
                sql_query = self._extract_sql(llm_output)
                clarifying = None

            self.logger.info(f"Generated SQL: {sql_query}")
            if clarifying:
                self.logger.info(f"Clarifying sub-question: {clarifying}")

            return sql_query, clarifying

        except Exception as e:
            self.logger.error(f"Error during NL to SQL translation: {e}")
            raise

    def _build_prompt(self, question: str, table_schema: str) -> str:
        """
        Construct a precise, context-rich prompt for the LLM.
        """
        prompt = (
            f"Given the following SQL table schema:\n"
            f"{table_schema}\n\n"
            f"Translate this natural language question into a safe, efficient SQL query.\n"
            f"Question: \"{question}\"\n"
            f"Instructions:\n"
            f"- Only use columns and tables present in the schema.\n"
            f"- If the question is ambiguous, return a clarifying sub-question instead of SQL.\n"
            f"- Do NOT generate any potentially unsafe or injection-prone queries.\n"
            f"- Output ONLY the SQL query (or clarifying sub-question), no explanation or commentary.\n"
            f"- Use parameterized queries or safe literal formatting when possible.\n"
        )
        return prompt

    def _extract_sql(self, llm_output: str) -> str:
        """
        Extract the SQL query from the LLM output.
        """
        lines = llm_output.strip().splitlines()
        sql_lines = []
        for line in lines:
            if line.strip().lower().startswith("select") or line.strip().lower().startswith("with"):
                sql_lines.append(line)
            elif sql_lines and not line.strip().startswith("--"):
                sql_lines.append(line)
        return "\n".join(sql_lines).strip("; \n")

    def _extract_clarifying_question(self, llm_output: str) -> str:
        """
        Extract a clarifying question from the LLM output.
        """
        # Typically, the LLM will output either a question or a sentence ending in '?'
        for line in llm_output.strip().splitlines():
            if line.strip().endswith("?"):
                return line.strip()
        # Fallback: return the first line
        return llm_output.strip().splitlines()[0]

# Example usage:
# logging.basicConfig(level=logging.INFO)
# converter = NLToSQLConverter(openai_api_key="sk-...")
# sql, clarification = converter.translate_nl_to_sql(
#     "Show me all users who signed up in the last 30 days.",
#     "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, signup_date DATE);"
# )
# if sql:
#     print("SQL Query:", sql)
# if clarification:
#     print("Clarifying Question:", clarification)