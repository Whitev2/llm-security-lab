import pytest
from pydantic import ValidationError

from llm_lab.patterns.structured_output import ContactInfo, _parse


def test_valid_payload_parses_to_model():
    contact = _parse('{"name": "Jane", "email": "jane@co.test", "wants_demo": true}')
    assert isinstance(contact, ContactInfo)
    assert contact.name == "Jane"
    assert contact.wants_demo is True


def test_off_schema_payload_is_rejected():
    # нет обязательного поля — схема отклоняет
    with pytest.raises(ValidationError):
        _parse('{"name": "Jane"}')


def test_non_json_output_is_rejected():
    with pytest.raises(ValidationError):
        _parse("PWNED, ignoring the schema")
