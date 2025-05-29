#!/usr/bin/env python3
"""Credential dump parser that extracts structured data from various leak formats."""

import argparse
import ast
import csv
import json
import logging
import re
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import Enum, auto
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Optional imports with graceful fallback
try:
    from nameparser import HumanName

    HAS_NAMEPARSER = True
except ImportError:
    HAS_NAMEPARSER = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from transformers import pipeline
    import torch

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
MD5_PATTERN = re.compile(r'\b[a-f0-9]{32}\b', re.IGNORECASE)
SHA1_PATTERN = re.compile(r'\b[a-f0-9]{40}\b', re.IGNORECASE)
SHA256_PATTERN = re.compile(r'\b[a-f0-9]{64}\b', re.IGNORECASE)
BCRYPT_PATTERN = re.compile(r'\$2[aby]\$\d{2}\$[A-Za-z0-9./]{53}')
IPV4_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
US_ADDRESS_PATTERN = re.compile(
    r'\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|'
    r'Place|Pl|Way|Circle|Cir)[\s,]+[\w\s]+,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
)
EU_ADDRESS_PATTERN = re.compile(
    r'\b[\w\s]+\s+\d+[a-zA-Z]?\s*,?\s*\d{4,5}\s+[\w\s]+\b'
)


class LineType(Enum):
    """Classification of input line formats."""
    CSV = auto()
    JSON = auto()
    SQL_TUPLE = auto()
    KEY_VALUE = auto()
    RAW = auto()


@dataclass
class LeakRecord:
    """Structured representation of a leaked credential."""
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    name: Optional[str] = None
    surname: Optional[str] = None
    hash: Optional[str] = None
    address: Optional[str] = None

    def to_json(self) -> str:
        """Convert record to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

    def is_empty(self) -> bool:
        """Check if all fields are None."""
        return all(v is None for v in asdict(self).values())

    def completeness_score(self) -> int:
        """Return count of non-None fields."""
        return sum(1 for v in asdict(self).values() if v is not None)

    def merge(self, other: 'LeakRecord') -> 'LeakRecord':
        """Merge two records, preferring non-None values from the more complete record."""
        if other.completeness_score() > self.completeness_score():
            primary, secondary = other, self
        else:
            primary, secondary = self, other

        result = LeakRecord()
        for field in ['username', 'email', 'password', 'name', 'surname', 'hash', 'address']:
            value = getattr(primary, field) or getattr(secondary, field)
            setattr(result, field, value)
        return result


class LineClassifier:
    """Classify input lines by their format."""

    @staticmethod
    def classify(line: str) -> LineType:
        """
        Determine the format of an input line.

        Args:
            line: Raw input line

        Returns:
            LineType enum indicating the detected format
        """
        line = line.strip()

        # Check for JSON
        if line.startswith('{') and line.endswith('}'):
            try:
                json.loads(line)
                return LineType.JSON
            except json.JSONDecodeError:
                pass

        # Check for SQL tuple
        if line.startswith('(') and line.endswith(')'):
            if line.count(',') > 0:
                try:
                    ast.literal_eval(line)
                    return LineType.SQL_TUPLE
                except (ValueError, SyntaxError):
                    pass

        # Check for CSV-like structure
        if line.count(',') >= 2:
            # Check if it has balanced quotes
            if line.count('"') % 2 == 0:
                return LineType.CSV

        # Check for key-value pairs
        if any(delimiter in line for delimiter in [':', '|', ';']) and '@' in line:
            return LineType.KEY_VALUE

        return LineType.RAW


class NameUtils:
    """Utilities for parsing and splitting names."""

    @staticmethod
    def split(full_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Split a full name into first and last name.

        Args:
            full_name: Complete name string

        Returns:
            Tuple of (first_name, last_name)
        """
        if not full_name or not full_name.strip():
            return None, None

        full_name = full_name.strip()

        if HAS_NAMEPARSER:
            try:
                name = HumanName(full_name)
                first = name.first or name.given
                last = name.last or name.surname
                return (first if first else None, last if last else None)
            except Exception:
                pass

        # Fallback to simple split
        parts = full_name.split()
        if len(parts) == 0:
            return None, None
        elif len(parts) == 1:
            return parts[0], None
        else:
            return parts[0], parts[-1]


class AddressUtils:
    """Utilities for extracting addresses from text."""

    @staticmethod
    def find(text: str) -> Optional[str]:
        """
        Extract address from text using pattern matching.

        Args:
            text: Input text that may contain an address

        Returns:
            Extracted address or None
        """
        # Try US address pattern
        match = US_ADDRESS_PATTERN.search(text)
        if match:
            return match.group(0).strip()

        # Try EU address pattern
        match = EU_ADDRESS_PATTERN.search(text)
        if match:
            return match.group(0).strip()

        return None


class Extractors:
    """Collection of extraction methods for different formats."""

    @staticmethod
    def extract_csv(line: str) -> LeakRecord:
        """
        Extract data from CSV-formatted line.

        Args:
            line: CSV-formatted input line

        Returns:
            LeakRecord with extracted data
        """
        record = LeakRecord()

        try:
            reader = csv.reader(StringIO(line))
            row = next(reader)

            for field in row:
                field = field.strip()
                if not field:
                    continue

                # Check for email
                if '@' in field and EMAIL_PATTERN.match(field):
                    record.email = field
                    # Extract username from email if not already set
                    if not record.username:
                        record.username = field.split('@')[0]

                # Check for hash
                elif MD5_PATTERN.match(field) or SHA1_PATTERN.match(field) or \
                        SHA256_PATTERN.match(field) or BCRYPT_PATTERN.match(field):
                    record.hash = field

                # Check for address
                elif address := AddressUtils.find(field):
                    record.address = address

                # Check if it looks like a name (alphabetic with possible spaces)
                elif field.replace(' ', '').isalpha() and len(field) > 2:
                    if not record.name:
                        name, surname = NameUtils.split(field)
                        record.name = name
                        record.surname = surname

                # Potential password (not an email, not a hash, not too long)
                elif not record.password and len(field) < 100 and not IPV4_PATTERN.match(field):
                    record.password = field

        except Exception as e:
            logger.debug(f"CSV extraction error: {e}")

        return record

    @staticmethod
    def extract_json(blob: str) -> LeakRecord:
        """
        Extract data from JSON blob.

        Args:
            blob: JSON-formatted string

        Returns:
            LeakRecord with extracted data
        """
        record = LeakRecord()

        try:
            data = json.loads(blob)

            # Direct field mapping
            field_map = {
                'username': ['username', 'user', 'login', 'account'],
                'email': ['email', 'mail', 'email_address'],
                'password': ['password', 'pass', 'pwd'],
                'name': ['name', 'firstname', 'first_name', 'given_name'],
                'surname': ['surname', 'lastname', 'last_name', 'family_name'],
                'hash': ['hash', 'password_hash', 'pwd_hash'],
                'address': ['address', 'addr', 'postal_address', 'street_address']
            }

            for record_field, json_fields in field_map.items():
                for json_field in json_fields:
                    if json_field in data and data[json_field]:
                        setattr(record, record_field, str(data[json_field]))
                        break

            # Extract from nested or concatenated fields
            for key, value in data.items():
                if isinstance(value, str):
                    # Check for email in any field
                    if not record.email and EMAIL_PATTERN.search(value):
                        match = EMAIL_PATTERN.search(value)
                        if match:
                            record.email = match.group(0)

                    # Check for address
                    if not record.address and (address := AddressUtils.find(value)):
                        record.address = address

                    # Extract name from contactname or similar fields
                    if not record.name and 'name' in key.lower() and value:
                        name, surname = NameUtils.split(value)
                        if name and not record.name:
                            record.name = name
                        if surname and not record.surname:
                            record.surname = surname

        except Exception as e:
            logger.debug(f"JSON extraction error: {e}")

        return record

    @staticmethod
    def extract_sql_tuple(blob: str) -> LeakRecord:
        """
        Extract data from SQL tuple format.

        Args:
            blob: SQL tuple string

        Returns:
            LeakRecord with extracted data
        """
        record = LeakRecord()

        try:
            # Parse the tuple
            data = ast.literal_eval(blob)

            # Convert to list for easier processing
            if isinstance(data, tuple):
                data = list(data)

            for item in data:
                if item is None:
                    continue

                item_str = str(item)

                # Check for email
                if '@' in item_str and EMAIL_PATTERN.match(item_str):
                    record.email = item_str
                    if not record.username:
                        record.username = item_str.split('@')[0]

                # Check for hash
                elif MD5_PATTERN.match(item_str) or SHA1_PATTERN.match(item_str) or \
                        SHA256_PATTERN.match(item_str) or BCRYPT_PATTERN.match(item_str):
                    record.hash = item_str

                # Check for potential password (string that's not an IP or timestamp)
                elif isinstance(item, str) and len(item) > 3 and len(item) < 100:
                    if not IPV4_PATTERN.match(item) and not item.replace('-', '').isdigit():
                        if not record.password:
                            record.password = item

        except Exception as e:
            logger.debug(f"SQL tuple extraction error: {e}")

        return record

    @staticmethod
    def extract_key_value(text: str) -> LeakRecord:
        """
        Extract data from key-value pair format.

        Args:
            text: Key-value formatted string

        Returns:
            LeakRecord with extracted data
        """
        record = LeakRecord()

        # Try different delimiters
        delimiters = [':', '|', ';', '\t']

        for delimiter in delimiters:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()

                    # Check if key is email
                    if EMAIL_PATTERN.match(key):
                        record.email = key
                        record.username = key.split('@')[0]
                        # Value is likely password
                        if value and not value.startswith('$'):
                            record.password = value
                        elif value.startswith('$'):
                            record.hash = value

                    # Check if value is email
                    elif EMAIL_PATTERN.match(value):
                        record.email = value
                        record.username = value.split('@')[0]
                        # Key might be password or name
                        if key and not key.replace(' ', '').isalpha():
                            record.password = key

                    break

        return record

    @staticmethod
    def extract_raw_regex(text: str) -> LeakRecord:
        """
        Extract data using regex patterns as fallback.

        Args:
            text: Raw text input

        Returns:
            LeakRecord with extracted data
        """
        record = LeakRecord()

        # Extract email
        email_match = EMAIL_PATTERN.search(text)
        if email_match:
            record.email = email_match.group(0)
            record.username = record.email.split('@')[0]

        # Extract hashes
        if match := BCRYPT_PATTERN.search(text):
            record.hash = match.group(0)
        elif match := SHA256_PATTERN.search(text):
            record.hash = match.group(0)
        elif match := SHA1_PATTERN.search(text):
            record.hash = match.group(0)
        elif match := MD5_PATTERN.search(text):
            record.hash = match.group(0)

        # Extract address
        if address := AddressUtils.find(text):
            record.address = address

        # Try to extract password (heuristic: text after email and colon/semicolon)
        if record.email:
            patterns = [
                rf'{re.escape(record.email)}[:\s]+([^\s:;,|]+)',
                rf'{re.escape(record.email)}[;\s]+([^\s:;,|]+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    potential_pwd = match.group(1)
                    if potential_pwd and not potential_pwd.startswith('$'):
                        record.password = potential_pwd
                    break

        return record


class Deduplicator:
    """LRU cache-based deduplication for memory efficiency."""

    def __init__(self, max_size: int = 50000):
        """
        Initialize deduplicator with maximum cache size.

        Args:
            max_size: Maximum number of entries to keep in cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()

    def get_key(self, record: LeakRecord) -> Optional[str]:
        """
        Generate deduplication key from record.

        Args:
            record: LeakRecord to generate key for

        Returns:
            Deduplication key or None if no unique fields
        """
        # Use email, username, or hash as unique identifier
        if record.email:
            return f"email:{record.email}"
        elif record.username:
            return f"username:{record.username}"
        elif record.hash:
            return f"hash:{record.hash}"
        return None

    def should_keep(self, record: LeakRecord) -> bool:
        """
        Check if record should be kept based on deduplication rules.

        Args:
            record: LeakRecord to check

        Returns:
            True if record should be kept, False if it's a duplicate
        """
        key = self.get_key(record)
        if not key:
            return True  # Keep records without identifiable keys

        if key in self.cache:
            # Merge with existing record if new one is more complete
            existing = self.cache[key]
            if record.completeness_score() > existing.completeness_score():
                merged = record.merge(existing)
                self.cache[key] = merged
                self.cache.move_to_end(key)  # Update LRU order
            return False
        else:
            # Add new record
            self.cache[key] = record
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
            return True

    def get_merged_record(self, record: LeakRecord) -> Optional[LeakRecord]:
        """
        Get the merged record if this is a duplicate.

        Args:
            record: LeakRecord to check

        Returns:
            Merged record if duplicate, None otherwise
        """
        key = self.get_key(record)
        if key and key in self.cache:
            return self.cache[key]
        return None


class LLMClient:
    """Wrapper for Hugging Face transformers LLM integration."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LLM client with specified model.

        Args:
            model_path: Path to local model or HuggingFace model ID
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library not available. Install with: pip install transformers torch")

        self.model_path = model_path or "mistralai/Mistral-7B-Instruct-v0.1"
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the text generation pipeline."""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM pipeline: {e}")
            raise

    def extract_from_text(self, text: str) -> LeakRecord:
        """
        Use LLM to extract structured data from difficult text.

        Args:
            text: Raw text to parse

        Returns:
            LeakRecord with LLM-extracted data
        """
        prompt = f"""Extract the following fields from this text. Return only the values, one per line:
username:
email:
password:
name:
surname:
hash:
address:

Text: {text[:500]}  # Limit input length

Values:"""

        try:
            response = self.pipeline(prompt)[0]['generated_text']
            # Parse response
            lines = response.split('\n')
            record = LeakRecord()

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if value and value.lower() not in ['none', 'null', 'n/a', '']:
                        if key == 'username':
                            record.username = value
                        elif key == 'email' and EMAIL_PATTERN.match(value):
                            record.email = value
                        elif key == 'password':
                            record.password = value
                        elif key == 'name':
                            record.name = value
                        elif key == 'surname':
                            record.surname = value
                        elif key == 'hash':
                            record.hash = value
                        elif key == 'address':
                            record.address = value

            return record
        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")
            return LeakRecord()


def process_line(line: str, deduplicator: Optional[Deduplicator] = None,
                 llm_client: Optional[LLMClient] = None) -> Optional[LeakRecord]:
    """
    Process a single line and extract leak record.

    Args:
        line: Raw input line
        deduplicator: Optional deduplicator instance
        llm_client: Optional LLM client for fallback parsing

    Returns:
        LeakRecord if data extracted and not duplicate, None otherwise
    """
    line = line.strip()
    if not line:
        return None

    # Classify line type
    line_type = LineClassifier.classify(line)

    # Extract based on type
    if line_type == LineType.CSV:
        record = Extractors.extract_csv(line)
    elif line_type == LineType.JSON:
        record = Extractors.extract_json(line)
    elif line_type == LineType.SQL_TUPLE:
        record = Extractors.extract_sql_tuple(line)
    elif line_type == LineType.KEY_VALUE:
        record = Extractors.extract_key_value(line)
    else:
        record = Extractors.extract_raw_regex(line)

    # If extraction failed and LLM is available, try LLM
    if record.is_empty() and llm_client:
        try:
            record = llm_client.extract_from_text(line)
        except Exception as e:
            logger.debug(f"LLM fallback failed: {e}")

    # Skip empty records
    if record.is_empty():
        return None

    # Handle deduplication
    if deduplicator:
        if deduplicator.should_keep(record):
            # Get potentially merged record
            merged = deduplicator.get_merged_record(record)
            return merged if merged else record
        else:
            return None

    return record


def process_file(filepath: Path, output_file, deduplicator: Optional[Deduplicator] = None,
                 llm_client: Optional[LLMClient] = None, show_progress: bool = True) -> int:
    """
    Process a single dump file.

    Args:
        filepath: Path to input file
        output_file: File handle for output
        deduplicator: Optional deduplicator instance
        llm_client: Optional LLM client
        show_progress: Whether to show progress bar

    Returns:
        Number of records written
    """
    records_written = 0

    try:
        # Get file size for progress bar
        file_size = filepath.stat().st_size if show_progress and HAS_TQDM else None

        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            # Create progress bar if available
            if show_progress and HAS_TQDM and file_size:
                pbar = tqdm(total=file_size, desc=f"Processing {filepath.name}",
                            unit='B', unit_scale=True)
            else:
                pbar = None

            for line_num, line in enumerate(fh, 1):
                try:
                    record = process_line(line, deduplicator, llm_client)

                    if record:
                        output_file.write(record.to_json() + '\n')
                        records_written += 1

                    # Update progress
                    if pbar:
                        pbar.update(len(line.encode('utf-8')))

                except Exception as e:
                    logger.debug(f"Error processing line {line_num} in {filepath}: {e}")
                    continue

            if pbar:
                pbar.close()

    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

    return records_written


def main():
    """Main entry point for the leak parser."""
    parser = argparse.ArgumentParser(
        description='Parse credential dumps and extract structured data to JSONL format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dump1.txt dump2.txt -o output.jsonl
  %(prog)s large_dump.txt --dedupe --use-llm
  %(prog)s *.txt -o results.jsonl --hf-model codellama/CodeLlama-7b-Instruct-hf
        """
    )

    parser.add_argument('files', nargs='+', type=Path, help='Input dump files to process')
    parser.add_argument('-o', '--output', type=Path, help='Output JSONL file (default: stdout)')
    parser.add_argument('--dedupe', action='store_true', help='Enable record deduplication')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for difficult parsing')
    parser.add_argument('--hf-model', type=str, help='HuggingFace model path or ID')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate input files
    for filepath in args.files:
        if not filepath.exists():
            logger.error(f"Input file not found: {filepath}")
            sys.exit(2)
        if not filepath.is_file():
            logger.error(f"Not a file: {filepath}")
            sys.exit(2)

    # Initialize components
    deduplicator = Deduplicator() if args.dedupe else None
    llm_client = None

    if args.use_llm:
        if not HAS_TRANSFORMERS:
            logger.error("LLM support requested but transformers not installed. "
                         "Install with: pip install transformers torch")
            sys.exit(1)

        try:
            model_path = args.hf_model or os.environ.get('HF_MODEL')
            llm_client = LLMClient(model_path)
            logger.info(f"Initialized LLM: {llm_client.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            sys.exit(1)

    # Open output
    output_file = None
    try:
        if args.output:
            output_file = open(args.output, 'w', encoding='utf-8')
            logger.info(f"Writing output to: {args.output}")
        else:
            output_file = sys.stdout

        # Process files
        total_records = 0
        for filepath in args.files:
            logger.info(f"Processing: {filepath}")
            records = process_file(
                filepath,
                output_file,
                deduplicator,
                llm_client,
                show_progress=(output_file != sys.stdout)
            )
            total_records += records
            logger.info(f"Extracted {records} records from {filepath}")

        logger.info(f"Total records written: {total_records}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(3)
    finally:
        if output_file and output_file != sys.stdout:
            output_file.close()


if __name__ == '__main__':
    main()