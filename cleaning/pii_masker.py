import pandas as pd
import re
import hashlib
import logging
from typing import Optional, List, Dict, Any, Callable
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# Regex patterns for PII detection (fallback or supplement to Presidio)
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4,6}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Default tokens for replacing PII types
PII_TOKENS = {
    "email": "<EMAIL_REDACTED>",
    "phone": "<PHONE_REDACTED>",
    "ssn": "<SSN_REDACTED>",
    "credit_card": "<CREDIT_CARD_REDACTED>",
    "ip_address": "<IP_REDACTED>",
}

class PIIMasker:
    """
    Detects and masks PII in pandas DataFrames with configurable strategies.

    Supported PII: emails, phone numbers, SSNs, credit cards, IP addresses.
    Masking: redact, hash, or replace with tokens.
    Uses regex and can utilize Microsoft Presidio if available.
    """

    SUPPORTED_PII = list(PII_PATTERNS.keys())
    SUPPORTED_MASKING = ["redact", "hash", "token"]

    def __init__(
        self,
        pii_types: Optional[List[str]] = None,
        masking_strategy: str = "redact",
        token_map: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
        presidio: bool = True,
        hash_algo: str = "sha256",
    ):
        """
        Args:
            pii_types: List of PII types to detect (defaults to all).
            masking_strategy: One of "redact", "hash", "token".
            token_map: Custom tokens for token replacement strategy.
            logger: Logger for logging and audit.
            presidio: Use Presidio if available (default: True).
            hash_algo: Hashing algorithm to use for 'hash' masking.
        """
        self.pii_types = pii_types or self.SUPPORTED_PII
        self.masking_strategy = masking_strategy
        self.token_map = {**PII_TOKENS, **(token_map or {})}
        self.logger = logger or logging.getLogger(__name__)
        self.use_presidio = presidio and PRESIDIO_AVAILABLE
        self.hash_algo = hash_algo
        if self.masking_strategy not in self.SUPPORTED_MASKING:
            raise ValueError(f"masking_strategy must be one of {self.SUPPORTED_MASKING}")
        self._setup_presidio()
        self.audit_log: List[Dict[str, Any]] = []

    def _setup_presidio(self):
        self.analyzer = None
        if self.use_presidio:
            self.analyzer = AnalyzerEngine()
            # Add custom recognizers for credit_card, ip_address if not present
            # Presidio already supports email, phone, ssn
            # Credit card
            if not any(r.entity_type == "CREDIT_CARD" for r in self.analyzer.get_recognizers()):
                self.analyzer.registry.add_recognizer(PatternRecognizer(
                    supported_entity="CREDIT_CARD",
                    patterns=[Pattern("credit_card", PII_PATTERNS["credit_card"].pattern, 0.7)],
                ))
            # IP address
            if not any(r.entity_type == "IP_ADDRESS" for r in self.analyzer.get_recognizers()):
                self.analyzer.registry.add_recognizer(PatternRecognizer(
                    supported_entity="IP_ADDRESS",
                    patterns=[Pattern("ip_address", PII_PATTERNS["ip_address"].pattern, 0.7)],
                ))

    def detect(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect PII in the DataFrame.

        Args:
            df: Input DataFrame.
            columns: Columns to check (default: object columns).

        Returns:
            Dict mapping column names to list of detected PII dicts.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        columns = columns or list(df.select_dtypes(include=["object", "string"]).columns)
        pii_report: Dict[str, List[Dict[str, Any]]] = {}

        for col in columns:
            results = []
            for idx, value in df[col].items():
                if pd.isnull(value):
                    continue
                matches = self._detect_value(str(value))
                for match in matches:
                    results.append({
                        "row": idx,
                        "pii_type": match["pii_type"],
                        "match": match["match"],
                        "start": match.get("start"),
                        "end": match.get("end"),
                    })
            if results:
                pii_report[col] = results
                self.logger.info(f"Detected {len(results)} PII values in column '{col}'")
        return pii_report

    def _detect_value(self, value: str) -> List[Dict[str, Any]]:
        matches = []
        if self.use_presidio and self.analyzer:
            presidio_result = self.analyzer.analyze(
                text=value, entities=[t.upper() for t in self.pii_types], language="en"
            )
            for res in presidio_result:
                matches.append({
                    "pii_type": res.entity_type.lower(),
                    "match": value[res.start:res.end],
                    "start": res.start,
                    "end": res.end,
                })
        else:
            for pii_type in self.pii_types:
                for m in PII_PATTERNS[pii_type].finditer(value):
                    matches.append({
                        "pii_type": pii_type,
                        "match": m.group(),
                        "start": m.start(),
                        "end": m.end(),
                    })
        return matches

    def mask(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        audit: bool = True
    ) -> pd.DataFrame:
        """
        Mask PII in the DataFrame according to the configured strategy.

        Args:
            df: Input DataFrame.
            columns: Columns to mask (default: object columns).
            audit: Whether to collect audit info (default: True).

        Returns:
            DataFrame with masked PII.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        columns = columns or list(df.select_dtypes(include=["object", "string"]).columns)
        df_masked = df.copy()
        audit_log = []

        for col in columns:
            def mask_func(val):
                if pd.isnull(val):
                    return val
                original_val = str(val)
                new_val, piis = self._mask_value(original_val)
                if audit and piis:
                    for pii in piis:
                        audit_log.append({
                            "column": col,
                            "original_value": original_val,
                            "masked_value": new_val,
                            "pii_type": pii["pii_type"],
                            "match": pii["match"],
                        })
                return new_val
            df_masked[col] = df_masked[col].apply(mask_func)

        if audit:
            self.audit_log = audit_log
            self.logger.info(f"PII masking complete. Masked {len(audit_log)} values.")
        return df_masked

    def _mask_value(self, value: str) -> (str, List[Dict[str, Any]]):
        """
        Mask all detected PII in a given string value.

        Returns:
            (masked_value, list of detected PII dicts)
        """
        matches = self._detect_value(value)
        masked_value = value
        # To ensure consistent masking, we'll mask each match in the order of appearance, non-overlapping
        for match in sorted(matches, key=lambda m: m["start"]):
            replacement = self._get_masked(match["pii_type"], match["match"])
            # To avoid overlapping replacements, rebuild string using indices
            masked_value = masked_value.replace(match["match"], replacement)
        return masked_value, matches

    def _get_masked(self, pii_type: str, match: str) -> str:
        if self.masking_strategy == "redact":
            return "[REDACTED]"
        elif self.masking_strategy == "token":
            return self.token_map.get(pii_type, f"<{pii_type.upper()}_REDACTED>")
        elif self.masking_strategy == "hash":
            h = hashlib.new(self.hash_algo)
            h.update(match.encode("utf-8"))
            return h.hexdigest()
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

    def audit_report(self) -> pd.DataFrame:
        """
        Return a compliance-friendly audit log of all masked PII.
        """
        return pd.DataFrame(self.audit_log or [])

    def get_supported_pii_types(self) -> List[str]:
        return self.SUPPORTED_PII

    def get_supported_masking_strategies(self) -> List[str]:
        return self.SUPPORTED_MASKING
