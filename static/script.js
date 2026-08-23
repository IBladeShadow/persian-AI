// ============================================================
// Persian AI — Chat Logic with Session Management
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const status = document.getElementById('status');
    const statusDot = status.querySelector('.status-dot');
    const statusText = status.querySelector('.status-text');
    const sidebar = document.getElementById('sidebar');
    const sidebarBtn = document.getElementById('sidebarBtn');
    const toggleSidebar = document.getElementById('toggleSidebar');
    const sessionsList = document.getElementById('sessionsList');
    const newChatBtn = document.getElementById('newChatBtn');
    const chatTitle = document.getElementById('chatTitle');

    let isProcessing = false;
    let currentSessionId = null;
    let messages = [];

    // ============================================================
    // Session Management
    // ============================================================

    async function loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            renderSessions(data.sessions);
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }

    function renderSessions(sessions) {
        sessionsList.innerHTML = '';
        sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = `session-item ${session.id === currentSessionId ? 'active' : ''}`;
            item.innerHTML = `
                <div class="session-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                </div>
                <div class="session-info">
                    <div class="session-title">${escapeHtml(session.title)}</div>
                    <div class="session-meta">${session.message_count} پیام</div>
                </div>
                <button class="session-delete" data-id="${session.id}" title="حذف">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 6L6 18M6 6l12 12"/>
                    </svg>
                </button>
            `;

            // Load session on click
            item.addEventListener('click', (e) => {
                if (!e.target.closest('.session-delete')) {
                    loadSession(session.id);
                }
            });

            // Delete session
            const deleteBtn = item.querySelector('.session-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteSession(session.id);
            });

            sessionsList.appendChild(item);
        });
    }

    async function createNewSession() {
        try {
            const response = await fetch('/api/sessions', { method: 'POST' });
            const data = await response.json();
            currentSessionId = data.session_id;
            messages = [];
            chatTitle.textContent = 'چت جدید';
            chatMessages.innerHTML = '';
            addWelcomeMessage();
            loadSessions();
            messageInput.focus();
        } catch (error) {
            console.error('Failed to create session:', error);
        }
    }

    async function loadSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            const data = await response.json();
            currentSessionId = sessionId;
            messages = data.messages || [];
            
            // Render messages
            chatMessages.innerHTML = '';
            if (messages.length === 0) {
                addWelcomeMessage();
            } else {
                messages.forEach(msg => {
                    if (msg.role === 'user') {
                        addUserMessageToDOM(msg.text, false);
                    } else {
                        addBotMessageToDOM(msg.text, false);
                    }
                });
                scrollToBottom();
            }

            // Update title
            if (messages.length > 0) {
                chatTitle.textContent = messages[0].text.substring(0, 30) + (messages[0].text.length > 30 ? '...' : '');
            } else {
                chatTitle.textContent = 'چت جدید';
            }

            // Update sidebar
            loadSessions();
            messageInput.focus();
        } catch (error) {
            console.error('Failed to load session:', error);
        }
    }

    async function saveCurrentSession() {
        if (!currentSessionId) return;
        try {
            await fetch(`/api/sessions/${currentSessionId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: messages,
                    title: messages.length > 0 ? messages[0].text.substring(0, 40) : 'چت جدید'
                })
            });
        } catch (error) {
            console.error('Failed to save session:', error);
        }
    }

    async function deleteSession(sessionId) {
        if (!confirm('آیا مطمئن هستید که می‌خواهید این چت را حذف کنید؟')) return;
        try {
            await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
            if (sessionId === currentSessionId) {
                currentSessionId = null;
                messages = [];
                chatMessages.innerHTML = '';
                addWelcomeMessage();
                chatTitle.textContent = 'چت جدید';
            }
            loadSessions();
            showToast('چت حذف شد');
        } catch (error) {
            console.error('Failed to delete session:', error);
        }
    }

    // ============================================================
    // Auto-resize textarea
    // ============================================================
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
        sendBtn.disabled = !messageInput.value.trim();
    });

    // ============================================================
    // Handle Enter key
    // ============================================================
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (messageInput.value.trim() && !isProcessing) {
                sendMessage();
            }
        }
    });

    // ============================================================
    // Send button click
    // ============================================================
    sendBtn.addEventListener('click', () => {
        if (messageInput.value.trim() && !isProcessing) {
            sendMessage();
        }
    });

    // ============================================================
    // Clear button
    // ============================================================
    clearBtn.addEventListener('click', () => {
        if (confirm('آیا مطمئن هستید که می‌خواهید چت را پاک کنید؟')) {
            chatMessages.innerHTML = '';
            messages = [];
            addWelcomeMessage();
            saveCurrentSession();
            showToast('چت پاک شد');
        }
    });

    // ============================================================
    // Copy button
    // ============================================================
    copyBtn.addEventListener('click', () => {
        const messageElements = chatMessages.querySelectorAll('.message-text');
        let text = '';
        messageElements.forEach(msg => {
            text += msg.textContent + '\n\n';
        });

        navigator.clipboard.writeText(text.trim()).then(() => {
            showToast('مکالمه کپی شد');
        }).catch(() => {
            const textarea = document.createElement('textarea');
            textarea.value = text.trim();
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            showToast('مکالمه کپی شد');
        });
    });

    // ============================================================
    // Sidebar toggle
    // ============================================================
    sidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('hidden');
        // On mobile, add overlay
        if (window.innerWidth <= 768) {
            const overlay = document.querySelector('.sidebar-overlay');
            if (!overlay) {
                const newOverlay = document.createElement('div');
                newOverlay.className = 'sidebar-overlay active';
                newOverlay.addEventListener('click', () => {
                    sidebar.classList.add('hidden');
                    newOverlay.classList.remove('active');
                });
                document.body.appendChild(newOverlay);
            }
        }
    });

    toggleSidebar.addEventListener('click', () => {
        sidebar.classList.add('hidden');
        const overlay = document.querySelector('.sidebar-overlay');
        if (overlay) overlay.classList.remove('active');
    });

    newChatBtn.addEventListener('click', createNewSession);

    // ============================================================
    // Send message
    // ============================================================
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Create session if needed
        if (!currentSessionId) {
            await createNewSession();
        }

        // Add user message
        addUserMessage(message);
        messages.push({ role: 'user', text: message });

        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Set processing state
        isProcessing = true;
        setStatus('thinking', 'در حال فکر کردن...');

        // Show typing indicator
        const typingIndicator = addTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message: message,
                    session_id: currentSessionId 
                })
            });

            const data = await response.json();

            // Remove typing indicator
            typingIndicator.remove();

            // Add bot response
            addBotMessage(data.response);
            messages.push({ role: 'bot', text: data.response });

            // Save session
            saveCurrentSession();

            // Update title if first message
            if (messages.length === 1) {
                chatTitle.textContent = message.substring(0, 30) + (message.length > 30 ? '...' : '');
                loadSessions();
            }

        } catch (error) {
            console.error('Error:', error);
            typingIndicator.remove();
            addBotMessage('خطا در برقراری ارتباط با سرور. لطفاً دوباره تلاش کنید.');
        } finally {
            isProcessing = false;
            setStatus('online', 'آنلاین');
            messageInput.focus();
        }
    }

    // ============================================================
    // Add welcome message
    // ============================================================
    function addWelcomeMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
                    <path d="M12 6a4 4 0 1 0 4 4 4 4 0 0 0-4-4z"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-sender">Persian AI</div>
                <div class="message-text">
                    سلام! خوش اومدی. من Persian AI هستم. با حافظه، ابزارهای داخلی، Retrieval روی دیتاست بزرگ و یک Transformer کوچک کار می‌کنم. پیامت رو بنویس :)
                </div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
    }

    // ============================================================
    // Add user message to chat
    // ============================================================
    function addUserMessage(text) {
        addUserMessageToDOM(text, true);
    }

    function addUserMessageToDOM(text, animate = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        if (!animate) messageDiv.style.animation = 'none';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-sender">شما</div>
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        if (animate) scrollToBottom();
    }

    // ============================================================
    // Add bot message to chat
    // ============================================================
    function addBotMessage(text) {
        addBotMessageToDOM(text, true);
    }

    function addBotMessageToDOM(text, animate = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        if (!animate) messageDiv.style.animation = 'none';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
                    <path d="M12 6a4 4 0 1 0 4 4 4 4 0 0 0-4-4z"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-sender">Persian AI</div>
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        if (animate) scrollToBottom();
    }

    // ============================================================
    // Add typing indicator
    // ============================================================
    function addTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
                    <path d="M12 6a4 4 0 1 0 4 4 4 4 0 0 0-4-4z"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-sender">Persian AI</div>
                <div class="message-text">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
        return messageDiv;
    }

    // ============================================================
    // Update status
    // ============================================================
    function setStatus(state, text) {
        status.className = 'status-badge ' + state;
        statusDot.className = 'status-dot ' + state;
        statusText.textContent = text;
    }

    // ============================================================
    // Scroll to bottom
    // ============================================================
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ============================================================
    // Escape HTML
    // ============================================================
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ============================================================
    // Show toast notification
    // ============================================================
    function showToast(message) {
        const existingToast = document.querySelector('.toast');
        if (existingToast) existingToast.remove();

        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    }

    // ============================================================
    // Initialize
    // ============================================================
    loadSessions();
    messageInput.focus();
});
