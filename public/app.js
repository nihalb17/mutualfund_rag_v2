/**
 * app.js - Phase 5
 * Frontend logic for the Mutual Fund RAG Chatbot.
 */

const CONFIG = {
    API_BASE: window.location.origin,
    SCHEME_ICONS: {
        'liquid': '⚡',
        'elss': '📈',
        'flexi': '🛡️'
    }
};

// Stateless - no sessionId required

// Elements
const chatFeed = document.getElementById('chat-feed');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const historyList = document.getElementById('history-list');

// Init
async function init() {
    // Session-less init

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });

    // Enter to send
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    sendBtn.addEventListener('click', handleSend);
    newChatBtn.addEventListener('click', resetSession);
}

// createNewSession removed (stateless)

async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';

    // Add User Message
    appendMessage('user', text);

    // Initial Bot Message element (empty, to be streamed into)
    const botMsgDiv = document.createElement('div');
    botMsgDiv.className = 'message bot-message';
    const msgContentDiv = document.createElement('div');
    msgContentDiv.className = 'msg-content';
    botMsgDiv.appendChild(msgContentDiv);
    chatFeed.appendChild(botMsgDiv);

    // Add Loading Indicator
    const loadingId = addLoadingIndicator();

    try {
        const response = await fetch(`${CONFIG.API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text
            })
        });

        if (!response.ok) {
            removeLoadingIndicator(loadingId);
            const errorData = await response.json().catch(() => ({}));
            let detail = errorData.detail;
            if (typeof detail === 'object') detail = JSON.stringify(detail);
            throw new Error(detail || `Server error (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);

                    if (data.chunk) {
                        removeLoadingIndicator(loadingId);
                        fullText += data.chunk;
                        msgContentDiv.innerHTML = formatText(fullText);
                        chatFeed.scrollTop = chatFeed.scrollHeight;
                    }

                    if (data.done) {
                        if (data.citations && data.citations.length > 0) {
                            let citationsHtml = `<div class="citations-list">`;
                            data.citations.forEach((url, index) => {
                                const domain = new URL(url).hostname.replace('groww.in', 'Groww');
                                citationsHtml += `
                                    <a href="${url}" target="_blank" class="citation-chip">
                                        <span class="citation-icon">🔗</span>
                                        <span class="citation-label">Source ${index + 1}: ${domain}</span>
                                    </a>
                                `;
                            });
                            citationsHtml += `</div>`;
                            botMsgDiv.insertAdjacentHTML('beforeend', citationsHtml);
                        }
                        addToHistory(text);
                    }

                    if (data.answer) {
                        // For guardrail or error case that returns full answer immediately
                        removeLoadingIndicator(loadingId);
                        msgContentDiv.innerHTML = formatText(data.answer);
                    }
                } catch (e) {
                    console.error("Error parsing stream chunk", e, line);
                }
            }
        }

    } catch (err) {
        removeLoadingIndicator(loadingId);
        console.error("Chat error:", err);
        msgContentDiv.innerHTML = `Error: ${err.message}. Please check if the server is running and your API key is valid.`;
    }
}

function appendMessage(role, text, citations = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;

    let html = `<div class="msg-content">${formatText(text)}</div>`;

    if (citations && citations.length > 0) {
        html += `<div class="citations-list">`;
        citations.forEach((url, index) => {
            const domain = new URL(url).hostname.replace('groww.in', 'Groww');
            html += `
                <a href="${url}" target="_blank" class="citation-chip">
                    <span class="citation-icon">🔗</span>
                    <span class="citation-label">Source ${index + 1}: ${domain}</span>
                </a>
            `;
        });
        html += `</div>`;
    }

    msgDiv.innerHTML = html;
    chatFeed.appendChild(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
}

function formatText(text) {
    // Basic markdown-like formatting for bold and lists
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/^\* (.*)/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
        .replace(/\n/g, '<br>');
}

function addLoadingIndicator() {
    const id = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = id;
    loadingDiv.className = 'message bot-message';
    loadingDiv.innerHTML = `
        <div class="loading">
            <span></span><span></span><span></span>
        </div>
    `;
    chatFeed.appendChild(loadingDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
    return id;
}

function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function addToHistory(text) {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerText = text;
    historyList.prepend(item);
}

async function resetSession() {
    if (confirm("Clear current chat history?")) {
        chatFeed.innerHTML = '';
        historyList.innerHTML = '';
        appendMessage('bot', "Chat cleared. How can I help you with Axis Mutual Funds today?");
    }
}

init();
