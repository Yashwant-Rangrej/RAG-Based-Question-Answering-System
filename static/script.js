/*
Nexus RAG Frontend logic
*/

const config = {
    apiKey: localStorage.getItem('nexus_api_key') || 'dev-secret-key-12345',
    currentDocId: null
};

// UI Elements
const elements = {
    health: document.getElementById('health-indicator'),
    docList: document.getElementById('doc-list'),
    uploadBtn: document.getElementById('upload-btn'),
    uploadModal: document.getElementById('upload-modal'),
    configBtn: document.getElementById('config-btn'),
    configModal: document.getElementById('config-modal'),
    apiKeyInput: document.getElementById('api-key-input'),
    saveConfigBtn: document.getElementById('save-config'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    progressContainer: document.getElementById('upload-progress'),
    progressBar: document.querySelector('.progress-bar'),
    chatForm: document.getElementById('chat-form'),
    userInput: document.getElementById('user-input'),
    chatMessages: document.getElementById('chat-messages'),
    sendBtn: document.getElementById('send-btn'),
    stopBtn: document.getElementById('stop-btn'),
    clearBtn: document.getElementById('clear-chat')
};

let currentAbortController = null;
let messageCounter = 0;

// Init
function init() {
    elements.apiKeyInput.value = config.apiKey;
    checkHealth();
    setInterval(checkHealth, 10000); // Check health every 10s

    // Event Listeners
    elements.uploadBtn.onclick = () => showModal(elements.uploadModal);
    elements.configBtn.onclick = () => showModal(elements.configModal);
    
    document.querySelectorAll('.close-modal').forEach(btn => {
        btn.onclick = (e) => hideModal(e.target.closest('.modal'));
    });

    elements.saveConfigBtn.onclick = saveConfig;

    // Drag & Drop
    elements.dropZone.onclick = () => elements.fileInput.click();
    elements.fileInput.onchange = (e) => handleFileUpload(e.target.files[0]);
    elements.dropZone.ondragover = (e) => { e.preventDefault(); elements.dropZone.classList.add('hover'); };
    elements.dropZone.ondragleave = () => elements.dropZone.classList.remove('hover');
    elements.dropZone.ondrop = (e) => { e.preventDefault(); handleFileUpload(e.dataTransfer.files[0]); };

    // Chat
    elements.chatForm.onsubmit = handleChatSubmit;
    elements.userInput.oninput = () => elements.sendBtn.disabled = !elements.userInput.value.trim();
    elements.clearBtn.onclick = clearChat;
    
    // Add document selection handler
    elements.docList.onclick = (e) => {
        const item = e.target.closest('.doc-item');
        if (item && item.classList.contains('ready')) {
            selectDocument(item.id.replace('doc-', ''));
        }
    };
    elements.stopBtn.onclick = () => {
        if (currentAbortController) {
            currentAbortController.abort();
            elements.stopBtn.classList.add('hidden');
            elements.sendBtn.classList.remove('hidden');
        }
    };

    // Suggestion chips
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.onclick = () => {
            elements.userInput.value = chip.innerText;
            elements.chatForm.dispatchEvent(new Event('submit'));
        };
    });
}

// API Interactions
async function apiRequest(endpoint, options = {}) {
    const headers = {
        'X-API-Key': config.apiKey,
        ...options.headers
    };

    try {
        const response = await fetch(endpoint, { ...options, headers });
        if (response.status === 401) {
            showModal(elements.configModal);
            throw new Error('Invalid API Key');
        }
        return response;
    } catch (err) {
        console.error(`API Error (${endpoint}):`, err);
        throw err;
    }
}

async function checkHealth() {
    try {
        const res = await fetch('/health');
        const data = await res.json();
        const indicator = elements.health;
        
        if (data.status === 'active') {
            indicator.className = 'status-badge ready';
            indicator.querySelector('.label').innerText = `Online (${data.ollama_ok ? 'Ollama OK' : 'Ollama Off'})`;
            elements.sendBtn.disabled = !elements.userInput.value.trim();
        } else {
            indicator.className = 'status-badge error';
            indicator.querySelector('.label').innerText = 'Offline';
        }
    } catch (e) {
        elements.health.className = 'status-badge error';
        elements.health.querySelector('.label').innerText = 'Connection Lost';
    }
}

async function handleFileUpload(file) {
    if (!file) return;
    
    elements.progressContainer.classList.remove('hidden');
    elements.progressBar.style.width = '20%';
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await apiRequest('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'Upload failed');
        }

        const data = await res.json();
        elements.progressBar.style.width = '100%';
        setTimeout(() => {
            hideModal(elements.uploadModal);
            elements.progressContainer.classList.add('hidden');
            elements.progressBar.style.width = '0%';
            addDocToList(data);
            pollStatus(data.document_id);
        }, 500);

    } catch (err) {
        alert('Upload Error: ' + err.message);
        elements.progressContainer.classList.add('hidden');
    }
}

async function pollStatus(docId) {
    const interval = setInterval(async () => {
        try {
            const res = await apiRequest(`/status/${docId}`);
            const data = await res.json();
            
            updateDocStatus(docId, data.status);
            
            if (data.status === 'ready' || data.status === 'failed') {
                clearInterval(interval);
                if (data.status === 'failed') alert(`Document processing failed: ${data.error}`);
            }
        } catch (e) {
            clearInterval(interval);
        }
    }, 2000);
}

async function handleChatSubmit(e) {
    e.preventDefault();
    const query = elements.userInput.value.trim();
    if (!query) return;

    // Add user message
    addMessage(query, 'user');
    elements.userInput.value = '';
    
    // Toggle buttons
    elements.sendBtn.classList.add('hidden');
    elements.stopBtn.classList.remove('hidden');
    
    currentAbortController = new AbortController();

    // Add loading AI message with Skeleton
    const msgId = addMessage('', 'ai');
    const msgDiv = document.getElementById(`msg-${msgId}`);
    const content = msgDiv.querySelector('.message-content');
    
    // Show Skeleton Loading
    content.innerHTML = `
        <div class="skeleton-container">
            <div class="skeleton skeleton-text"></div>
            <div class="skeleton skeleton-text"></div>
            <div class="skeleton skeleton-text short"></div>
        </div>
    `;

    try {
        const response = await apiRequest('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query, 
                top_k: 5,
                document_id: config.currentDocId 
            }),
            signal: currentAbortController.signal
        });

        if (!response.ok) throw new Error('Query failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = '';
        let hasSources = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.replace('data: ', '').trim();
                if (jsonStr === '[DONE]') break;

                try {
                    const data = JSON.parse(jsonStr);
                    
                    // Handle sources (first chunk)
                    if (data.sources && !hasSources) {
                        content.innerHTML = ''; // Remove skeleton
                        const answerText = document.createElement('div');
                        answerText.className = 'text';
                        content.appendChild(answerText);
                        hasSources = true;
                    }

                    // Handle individual tokens
                    if (data.token) {
                        if (!hasSources) {
                             content.innerHTML = '<div class="text"></div>';
                             hasSources = true;
                        }
                        const textDiv = content.querySelector('.text');
                        fullAnswer += data.token;
                        textDiv.innerHTML = fullAnswer.replace(/\n/g, '<br>');
                        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
                    }
                    
                    // Handle direct answer (non-streaming fallback)
                    if (data.answer) {
                        content.innerHTML = `<div class="text">${data.answer.replace(/\n/g, '<br>')}</div>`;
                    }

                } catch (e) {
                    console.error('Error parsing stream chunk', e);
                }
            }
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Generation aborted by user');
            if (content.querySelector('.skeleton-container')) {
                content.innerHTML = '<div class="text muted">Generation stopped.</div>';
            }
        } else {
            content.innerHTML = `<div class="text error">Error: ${err.message}</div>`;
        }
    } finally {
        elements.sendBtn.classList.remove('hidden');
        elements.stopBtn.classList.add('hidden');
        elements.sendBtn.disabled = !elements.userInput.value.trim();
        currentAbortController = null;
    }
}



function clearChat() {
    elements.chatMessages.innerHTML = '';
    // Add back the greeting
    elements.chatMessages.innerHTML = `
        <div class="message ai greeting">
            <div class="message-content">
                <h3>Welcome to Nexus RAG</h3>
                <p>I can help you analyze and query your documents with factual precision. Upload a PDF or TXT to get started.</p>
            </div>
        </div>
    `;
}

// UI Helpers
function showModal(modal) {
    if (!modal) return;
    modal.style.display = 'flex';
}

function hideModal(modal) {
    if (!modal) return;
    modal.style.display = 'none';
}

function saveConfig() {
    const key = elements.apiKeyInput.value.trim();
    if (!key) return alert('API Key is required');
    
    config.apiKey = key;
    localStorage.setItem('nexus_api_key', key);
    hideModal(elements.configModal);
    alert('Settings saved');
}

function addDocToList(doc) {
    const emptyState = elements.docList.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const docDiv = document.createElement('div');
    docDiv.className = 'doc-item';
    docDiv.id = `doc-${doc.document_id}`;
    docDiv.innerHTML = `
        <div class="name">${doc.filename}</div>
        <span class="status">${doc.status}</span>
    `;
    elements.docList.prepend(docDiv);
}

function updateDocStatus(docId, status) {
    const docDiv = document.getElementById(`doc-${docId}`);
    if (docDiv) {
        const statusSpan = docDiv.querySelector('.status');
        statusSpan.innerText = status;
        if (status === 'ready') {
            docDiv.classList.add('ready');
            // If no document selected, select the first ready one (latest)
            if (!config.currentDocId) selectDocument(docId);
        }
        if (status === 'failed') docDiv.classList.add('error');
    }
}

function selectDocument(docId) {
    // Unselect previous
    document.querySelectorAll('.doc-item').forEach(el => el.classList.remove('active'));
    
    const docDiv = document.getElementById(`doc-${docId}`);
    if (docDiv) {
        docDiv.classList.add('active');
        config.currentDocId = docId;
        const filename = docDiv.querySelector('.name').innerText;
        document.getElementById('current-context').innerText = `Context: ${filename}`;
    } else {
        config.currentDocId = null;
        document.getElementById('current-context').innerText = 'Context: Full Library';
    }
}

function addMessage(content, role) {
    const id = ++messageCounter;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = `msg-${id}`;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="text">${content.replace(/\n/g, '<br>')}</div>
        </div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    return id;
}

init();
