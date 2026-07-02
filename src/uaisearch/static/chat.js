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

async function askQuestion(query) {
    const { bubble, textNode } = createAnswerBubble();
    const params = new URLSearchParams({ query, conversation_id: conversationId });
    const response = await fetch(`/api/v1/answer?${params}`, { method: "POST" });
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
            }
        }
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
