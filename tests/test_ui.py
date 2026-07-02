from fastapi.testclient import TestClient

from uaisearch.api import app


def test_chat_page_serves_html_shell():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="chat-form"' in response.text
    assert 'id="chat-log"' in response.text
