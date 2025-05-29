"""Test suite for leak_parser.py"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from leak_parser import (
    AddressUtils,
    Deduplicator,
    Extractors,
    LeakRecord,
    LineClassifier,
    LineType,
    NameUtils,
)


class TestLeakRecord:
    """Test LeakRecord dataclass."""

    def test_to_json(self):
        """Test JSON serialization."""
        record = LeakRecord(username="john", email="john@example.com")
        json_str = record.to_json()
        data = json.loads(json_str)
        assert data["username"] == "john"
        assert data["email"] == "john@example.com"
        assert data["password"] is None

    def test_is_empty(self):
        """Test empty record detection."""
        assert LeakRecord().is_empty()
        assert not LeakRecord(username="test").is_empty()

    def test_completeness_score(self):
        """Test completeness scoring."""
        assert LeakRecord().completeness_score() == 0
        assert LeakRecord(username="test").completeness_score() == 1
        assert LeakRecord(username="test", email="test@example.com").completeness_score() == 2

    def test_merge(self):
        """Test record merging."""
        r1 = LeakRecord(username="john", email="john@example.com")
        r2 = LeakRecord(username="john", password="secret123")
        merged = r1.merge(r2)
        assert merged.username == "john"
        assert merged.email == "john@example.com"
        assert merged.password == "secret123"


class TestLineClassifier:
    """Test line classification."""

    def test_classify_json(self):
        """Test JSON line detection."""
        assert LineClassifier.classify('{"email": "test@example.com"}') == LineType.JSON
        assert LineClassifier.classify('{"invalid": json}') != LineType.JSON

    def test_classify_sql_tuple(self):
        """Test SQL tuple detection."""
        assert LineClassifier.classify("(1, 'john', 'john@example.com')") == LineType.SQL_TUPLE
        assert LineClassifier.classify("(single)") != LineType.SQL_TUPLE

    def test_classify_csv(self):
        """Test CSV detection."""
        assert LineClassifier.classify("john,doe,john@example.com,password") == LineType.CSV
        assert LineClassifier.classify('john,"doe, jr",john@example.com') == LineType.CSV

    def test_classify_key_value(self):
        """Test key-value detection."""
        assert LineClassifier.classify("john@example.com:password123") == LineType.KEY_VALUE
        assert LineClassifier.classify("john@example.com|password123") == LineType.KEY_VALUE

    def test_classify_raw(self):
        """Test raw/fallback classification."""
        assert LineClassifier.classify("random text") == LineType.RAW
        assert LineClassifier.classify("") == LineType.RAW


class TestExtractors:
    """Test extraction methods."""

    def test_extract_csv(self):
        """Test CSV extraction."""
        line = "john.doe@example.com,John Doe,password123,5f4dcc3b5aa765d61d8327deb882cf99"
        record = Extractors.extract_csv(line)
        assert record.email == "john.doe@example.com"
        assert record.username == "john.doe"
        assert record.name == "John"
        assert record.surname == "Doe"
        assert record.password == "password123"
        assert record.hash == "5f4dcc3b5aa765d61d8327deb882cf99"

    def test_extract_json(self):
        """Test JSON extraction."""
        blob = '{"email": "test@example.com", "password": "secret", "firstname": "Test"}'
        record = Extractors.extract_json(blob)
        assert record.email == "test@example.com"
        assert record.password == "secret"
        assert record.name == "Test"

    def test_extract_sql_tuple(self):
        """Test SQL tuple extraction."""
        blob = "(123, 'user@example.com', 'hashedpassword', '2023-01-01')"
        record = Extractors.extract_sql_tuple(blob)
        assert record.email == "user@example.com"
        assert record.username == "user"
        assert record.password == "hashedpassword"

    def test_extract_key_value(self):
        """Test key-value extraction."""
        text = "admin@site.com:SuperSecret123"
        record = Extractors.extract_key_value(text)
        assert record.email == "admin@site.com"
        assert record.username == "admin"
        assert record.password == "SuperSecret123"

    def test_extract_raw_regex(self):
        """Test regex fallback extraction."""
        text = "User john.smith@company.com has password: MyPass123!"
        record = Extractors.extract_raw_regex(text)
        assert record.email == "john.smith@company.com"
        assert record.username == "john.smith"
        assert record.password == "MyPass123!"


class TestNameUtils:
    """Test name parsing utilities."""

    def test_split_simple(self):
        """Test simple name splitting."""
        assert NameUtils.split("John Doe") == ("John", "Doe")
        assert NameUtils.split("John") == ("John", None)
        assert NameUtils.split("") == (None, None)

    def test_split_complex(self):
        """Test complex name splitting."""
        # Without nameparser, falls back to simple split
        assert NameUtils.split("John Michael Doe") == ("John", "Doe")


class TestAddressUtils:
    """Test address extraction."""

    def test_us_address(self):
        """Test US address pattern."""
        text = "Located at 123 Main Street, Springfield, IL 62701"
        assert AddressUtils.find(text) == "123 Main Street, Springfield, IL 62701"

    def test_eu_address(self):
        """Test EU address pattern."""
        text = "Send to Hauptstraße 42, 10115 Berlin"
        address = AddressUtils.find(text)
        assert address is not None
        assert "42" in address

    def test_no_address(self):
        """Test when no address present."""
        assert AddressUtils.find("No address here") is None


class TestDeduplicator:
    """Test deduplication functionality."""

    def test_deduplication_by_email(self):
        """Test deduplication using email."""
        dedup = Deduplicator(max_size=10)
        r1 = LeakRecord(email="test@example.com", password="pass1")
        r2 = LeakRecord(email="test@example.com", password="pass2", username="test")

        assert dedup.should_keep(r1) is True
        assert dedup.should_keep(r2) is False

        # Check that more complete record is kept
        merged = dedup.get_merged_record(r2)
        assert merged.username == "test"

    def test_deduplication_by_username(self):
        """Test deduplication using username."""
        dedup = Deduplicator(max_size=10)
        r1 = LeakRecord(username="johndoe")
        r2 = LeakRecord(username="johndoe", email="john@example.com")

        assert dedup.should_keep(r1) is True
        assert dedup.should_keep(r2) is False

    def test_lru_eviction(self):
        """Test LRU cache eviction."""
        dedup = Deduplicator(max_size=2)
        r1 = LeakRecord(email="user1@example.com")
        r2 = LeakRecord(email="user2@example.com")
        r3 = LeakRecord(email="user3@example.com")

        assert dedup.should_keep(r1) is True
        assert dedup.should_keep(r2) is True
        assert dedup.should_keep(r3) is True

        # r1 should have been evicted
        assert "email:user1@example.com" not in dedup.cache
        assert "email:user3@example.com" in dedup.cache