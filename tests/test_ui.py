from fastapi.testclient import TestClient

from uaisearch.api import app


def test_chat_page_serves_html_shell():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="chat-form"' in response.text
    assert 'id="chat-log"' in response.text


def test_chat_js_streams_tokens_via_fetch_reader():
    client = TestClient(app)
    response = client.get("/static/chat.js")
    assert response.status_code == 200
    assert "getReader" in response.text
    assert "conversation_id" in response.text
    assert "askQuestion" in response.text


def test_chat_js_renders_sources_and_related_question_chips_on_final_event():
    client = TestClient(app)
    response = client.get("/static/chat.js")
    assert "renderFinalEvent" in response.text
    assert "related_questions" in response.text
    assert "askQuestion(question)" in response.text


def test_chat_js_handles_errors():
    client = TestClient(app)
    response = client.get("/static/chat.js")
    assert "response.ok" in response.text or "catch" in response.text


def test_chat_js_renders_numbered_clickable_source_links_safely():
    client = TestClient(app)
    response = client.get("/static/chat.js")
    assert 'createElement("a")' in response.text
    assert "noopener noreferrer" in response.text
    assert "textContent" in response.text
    assert "innerHTML" not in response.text
