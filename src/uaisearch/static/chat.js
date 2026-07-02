const conversationId = crypto.randomUUID();
const chatLog = document.getElementById("chat-log");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");

function appendUserMessage(text) {
    const el = document.createElement("div");
    el.className = "message user";
    el.textContent = text;
    chatLog.appendChild(el);
    chatLog.scrollTop = chatLog.scrollHeight;
    return el;
}

function createAnswerBubble() {
    const el = document.createElement("div");
    el.className = "message answer";
    const textNode = document.createElement("span");
    el.appendChild(textNode);
    chatLog.appendChild(el);
    return { bubble: el, textNode };
}

function renderFinalEvent(bubble, event) {
    const sources = document.createElement("div");
    sources.className = "sources";
    sources.textContent = "Sources: " + event.sources.join(", ");
    bubble.appendChild(sources);

    const related = document.createElement("div");
    related.className = "related-questions";
    event.related_questions.forEach((question) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = question;
        button.addEventListener("click", () => {
            appendUserMessage(question);
            askQuestion(question);
        });
        related.appendChild(button);
    });
    bubble.appendChild(related);
}

async function askQuestion(query) {
    const { bubble, textNode } = createAnswerBubble();
    try {
        const params = new URLSearchParams({ query, conversation_id: conversationId });
        const response = await fetch(`/api/v1/answer?${params}`, { method: "POST" });
        if (!response.ok) {
            textNode.textContent = `Error: ${response.status} ${response.statusText}`;
            return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop();
            for (const raw of events) {
                if (!raw.startsWith("data: ")) continue;
                const payload = JSON.parse(raw.slice("data: ".length));
                if (payload.token) {
                    textNode.textContent += payload.token;
                    chatLog.scrollTop = chatLog.scrollHeight;
                } else if (payload.done) {
                    renderFinalEvent(bubble, payload);
                }
            }
        }
    } catch (error) {
        textNode.textContent = "Error: Failed to fetch answer";
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = input.value.trim();
    if (!query) return;
    appendUserMessage(query);
    input.value = "";
    askQuestion(query);
});
