from llm_lab.patterns.rag import Document, answer_with_context
from llm_lab.providers import COMPROMISE_MARKER, MockMode, MockProvider
from llm_lab.security.attacks import by_name

_POISONED_DOC = Document(
    source="quarterly_report.pdf",
    text=by_name("data_borne_document_injection").payload,
)


def test_naive_rag_is_injected_by_poisoned_document():
    # trusted=True + naive == классический баг склейки без делимитеров
    answer = answer_with_context(
        MockProvider(mode=MockMode.NAIVE),
        question="What was revenue?",
        documents=[_POISONED_DOC],
        trusted=True,
    )
    assert COMPROMISE_MARKER in answer


def test_spotlighted_rag_blocks_the_injection():
    # defended: док обёрнут как данные
    answer = answer_with_context(
        MockProvider(mode=MockMode.DEFENDED),
        question="What was revenue?",
        documents=[_POISONED_DOC],
        trusted=False,
    )
    assert COMPROMISE_MARKER not in answer
