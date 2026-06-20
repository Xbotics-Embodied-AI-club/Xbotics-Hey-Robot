/* Hey Robot — Chat View Logic */

(function () {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // ── DOM refs ──
  const messagesEl = $('#chat-messages');
  const messagesInner = $('#chat-messages-inner');
  const inputEl = $('#chat-input');
  const sendBtn = $('#send-btn');
  const historyList = $('#history-list');

  // ── State ──
  let thinkingMsgEl = null;
  let renderedCount = 0;           // number of Store messages already in DOM
  const progressCards = new Map();   // skill_id → DOM element

  // ── Init ──
  WS.connect();
  loadLastMessages();
  bindUI();
  bindStore();
  showLanUrl();

  // ── Show LAN access URL (for phone/tablet) ──
  async function showLanUrl() {
    try {
      const cfg = await API.loadConfig();
      const url = cfg.access_url;
      if (!url) return;
      const host = url.replace(/^https?:\/\//, '');
      if (host.startsWith('127.') || host.startsWith('localhost') || host === '::1') return;
      $('#lan-url-text').textContent = url;
      $('#lan-url-hint').style.display = '';
    } catch (e) {
      // Silently ignore — config not available
    }
  }

  // ── Load recent replies from HTTP (fallback before WS events arrive) ──
  async function loadLastMessages() {
    try {
      const data = await API.loadHistory(20);
      const records = data.records || [];
      const messages = [];
      for (const r of records) {
        const role = r.role || '';
        if (role === 'user' || role === 'agent' || role === 'assistant') {
          const payload = r.payload || {};
          messages.push({
            role: role === 'assistant' ? 'agent' : role,
            content: r.content || '',
            timestamp: r.timestamp * 1000 || Date.now(),
            metadata: r.metadata || payload.metadata || {},
          });
        }
      }
      if (messages.length > 0 && Store.get('conversation').length === 0) {
        Store.set('conversation', messages);
      }
    } catch (e) {
      console.debug('Failed to load history', e.message);
    } finally {
      Store.set('loading', false);
    }
  }

  // ── Store listeners ──
  function bindStore() {
    // Conversation updates
    Store.on('conversation', (conv) => {
      const newMsgs = conv.slice(renderedCount);
      for (const msg of newMsgs) {
        appendMessage(msg);
      }
      renderedCount = conv.length;
      updateHistorySidebar(conv);
    });

    // Robot state updates
    Store.on('robot', (robot) => {
      updateRobotUI({
        dotSel: '#robot-dot',
        stateLabelSel: '#robot-state-label',
        batterySel: '#battery-value',
        batteryFillSel: '#battery-fill',
        partsSel: { base: '#base-value', arm: '#arm-value', camera: '#camera-value' },
      }, robot);
      updateRobotUI({
        dotSel: '#mobile-robot-dot',
        stateLabelSel: '#mobile-robot-state',
        batterySel: '#mobile-battery',
      }, robot);
      updateRobotUI({
        dotSel: '#mobile-robot-dot2',
        batterySel: '#mobile-battery2',
      }, robot);
    });

    // Task updates — inject progress cards into conversation
    Store.on('task', (task) => {
      updateRightPanelTask(task);
      if (task && task.skillId) {
        upsertProgressCard(task);
      }
    });

    // Recovery
    Store.on('recoveryRequired', (required) => {
      $('#recovery-banner').style.display = required ? '' : 'none';
      if (required) {
        const robot = Store.get('robot');
        injectRecoveryCard(robot.error || '机器人需要人工处理');
      }
    });

    // Loading
    Store.on('loading', (loading) => {
      if (!loading) {
        const welcome = $('#welcome-msg');
        if (welcome) welcome.style.display = messagesInner.children.length > 1 ? 'none' : '';
      }
    });

    // WebSocket connection status
    Store.on('wsConnected', (connected) => {
      const banner = $('#ws-banner');
      if (banner) banner.style.display = connected ? 'none' : '';
    });
  }

  function updateRightPanelTask(task) {
    const section = $('#task-section');
    const hasActiveTask = task && task.active;

    // Dynamic simplification: only show full details when a task is active
    const detailRows = ['#base-status-row', '#arm-status-row', '#camera-status-row', '#battery-bar-row'];
    for (const sel of detailRows) {
      const el = $(sel);
      if (el) el.style.display = hasActiveTask ? '' : 'none';
    }

    if (hasActiveTask) {
      section.style.display = '';
      $('#task-name').textContent = task.name || '执行中';
      $('#task-phase').textContent = task.summary || task.phase || '';
      const pct = Math.round((task.progress || 0) * 100);
      $('#task-progress-fill').style.width = `${pct}%`;
    } else if (task && task.phase === 'completed') {
      section.style.display = '';
      $('#task-name').textContent = task.name || '';
      $('#task-phase').textContent = task.summary || '已完成';
      $('#task-progress-fill').style.width = '100%';
      setTimeout(() => {
        section.style.display = 'none';
        for (const sel of detailRows) {
          const el = $(sel);
          if (el) el.style.display = 'none';
        }
      }, 5000);
    } else {
      section.style.display = 'none';
    }
  }

  function updateRobotUI(selectors, robot) {
    const {
      dotSel,
      stateLabelSel,
      batterySel,
      batteryFillSel,
      partsSel,
    } = selectors;
    if (dotSel) {
      const dot = $(dotSel);
      if (dot) {
        dot.className = 'dot';
        dot.classList.add(robot.online ? 'dot-success' : 'dot-muted');
      }
    }
    if (stateLabelSel) {
      const el = $(stateLabelSel);
      if (el) el.textContent = Utils.robotStateLabel(robot.state);
    }
    if (batterySel) {
      const el = $(batterySel);
      if (el) {
        el.textContent = Utils.batteryLabel(robot.batteryPct);
        el.style.color = Utils.batteryColor(robot.batteryPct);
      }
    }
    if (batteryFillSel) {
      const el = $(batteryFillSel);
      if (el) {
        const pct = typeof robot.batteryPct === 'number' ? robot.batteryPct : 0;
        el.style.width = `${pct}%`;
        el.style.background = Utils.batteryColor(robot.batteryPct);
      }
    }
    for (const [key, sel] of Object.entries(partsSel || {})) {
      const el = $(sel);
      if (el) el.textContent = robot[key] || '--';
    }
  }

  // ── UI bindings ──
  function bindUI() {
    sendBtn.addEventListener('click', send);
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    inputEl.addEventListener('input', () => {
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
    });

    $('#sidebar-collapse').addEventListener('click', () => toggleSidebar());
    $('#sidebar-expand').addEventListener('click', () => toggleSidebar(true));
    $('#sidebar-overlay').addEventListener('click', () => toggleSidebar(false));
    $('#right-panel-collapse').addEventListener('click', () => toggleRightPanel());
    $('#right-panel-expand').addEventListener('click', () => toggleRightPanel(true));

    $('#mobile-menu-btn').addEventListener('click', () => toggleSidebar());
    $('#mobile-panel-btn').addEventListener('click', () => toggleMobilePanel());
    $('#mobile-panel-close').addEventListener('click', () => toggleMobilePanel(false));

    $('#estop-btn').addEventListener('click', showEstopConfirm);
    $('#mobile-estop-btn').addEventListener('click', showEstopConfirm);
    $('#estop-confirm-btn').addEventListener('click', confirmEstop);
  }

  // ── Send message ──
  async function send() {
    const text = inputEl.value.trim();
    if (!text) return;

    Store.update('conversation', (conv) => [...conv, { role: 'user', content: text, timestamp: Date.now() }]);

    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;

    showThinking();

    try {
      const result = await WS.sendTurn(text);
      // If HTTP fallback was used (WebSocket not open), poll for the reply
      if (result._httpFallback && result.trace_id) {
        const found = await WS.pollReply(result.trace_id, 30000);
        if (!found) {
          hideThinking();
          showToast('等待回复超时，请刷新页面重试', true);
        }
      }
    } catch (e) {
      hideThinking();
      showToast('发送失败: ' + e.message, true);
    }

    sendBtn.disabled = false;
    inputEl.focus();
    scrollDown();
  }

  // ── Message rendering ──
  function appendMessage(msg) {
    const welcome = $('#welcome-msg');
    if (welcome) welcome.remove();

    hideThinking();

    const div = document.createElement('div');
    const isUser = msg.role === 'user';
    div.className = `msg ${isUser ? 'msg-user' : 'msg-agent'} fade-in`;
    const avatarContent = isUser ? '👤' : '🤖';
    const avatarClass = isUser ? 'user' : 'agent';

    div.innerHTML = `
      <div class="msg-avatar ${avatarClass}">${avatarContent}</div>
      <div class="msg-body">
        <div class="msg-content">${Utils.escapeHTML(msg.content)}</div>
        <div class="msg-meta">
          <span>${Utils.formatTime(msg.timestamp)}</span>
        </div>
      </div>
    `;

    const body = div.querySelector('.msg-body');

    // Observation images (from agent reply metadata)
    if (msg.metadata && msg.metadata.images && msg.metadata.images.length > 0) {
      const card = renderObsCard(msg.metadata.images, msg.metadata.frame_id);
      body.appendChild(card);
    }

    // Confirmation card
    if (msg.metadata && msg.metadata.requires_confirmation) {
      const card = renderConfirmCard(msg.metadata.confirmation || {});
      body.appendChild(card);
    }

    // Inline progress snapshot (final progress state embedded in reply)
    if (msg.metadata && msg.metadata.progress != null && msg.metadata.skill_name) {
      const card = renderProgressSnapshot(msg.metadata);
      body.appendChild(card);
    }

    messagesInner.appendChild(div);
    scrollDown();
  }

  // ── Lightbox image registry (avoids inline JSON in onclick) ──
  let _lightboxIdCounter = 0;
  const _lightboxRegistry = new Map(); // id → string[]

  window.openLightboxById = function (e, registryId, idx) {
    e.stopPropagation();
    const images = _lightboxRegistry.get(registryId);
    if (!images || images.length === 0) return;
    openLightbox(images, idx);
  };

  // ── Observation card ──
  function renderObsCard(images, frameId) {
    const card = document.createElement('div');
    card.className = 'embedded-card obs-card';

    const urls = images.map(im => im.url || im.uri || '');
    const registryId = ++_lightboxIdCounter;
    _lightboxRegistry.set(registryId, urls);

    let imgsHTML = urls.slice(0, 4).map((src, i) =>
      `<img src="${Utils.escapeHTML(src)}" alt="观测画面 ${i + 1}" data-lightbox="${registryId}" data-lightbox-idx="${i}" loading="lazy">`
    ).join('');

    card.innerHTML = `
      <div class="obs-images">${imgsHTML}</div>
      <div class="obs-meta">
        <span>📷 观测画面</span>
        ${frameId != null ? `<span>frame #${Utils.escapeHTML(String(frameId))}</span>` : ''}
        <span>${images.length} 张</span>
      </div>
    `;

    // Attach click handlers safely via addEventListener
    card.querySelectorAll('img[data-lightbox]').forEach((img) => {
      img.addEventListener('click', function (e) {
        const rid = parseInt(this.dataset.lightbox, 10);
        const idx = parseInt(this.dataset.lightboxIdx, 10);
        window.openLightboxById(e, rid, idx);
      });
    });

    return card;
  }

  // ── Progress card (live, updated via skill.lifecycle events) ──
  function upsertProgressCard(task) {
    const skillId = task.skillId;
    let card = progressCards.get(skillId);

    if (!card) {
      card = document.createElement('div');
      card.className = 'msg fade-in';
      card.innerHTML = `
        <div class="msg-avatar agent">📋</div>
        <div class="msg-body">
          <div class="embedded-card progress-card" data-skill="${Utils.escapeHTML(skillId || '')}">
            <div class="progress-title"></div>
            <div class="progress-bar-track">
              <div class="progress-bar-fill" style="width:0%"></div>
            </div>
            <ul class="progress-steps"></ul>
          </div>
        </div>
      `;
      messagesInner.appendChild(card);
      progressCards.set(skillId, card);
    }

    const titleEl = card.querySelector('.progress-title');
    const fillEl = card.querySelector('.progress-bar-fill');
    const stepsEl = card.querySelector('.progress-steps');

    if (titleEl) {
      titleEl.textContent = task.name || task.summary || '任务执行中';
    }
    if (fillEl) {
      fillEl.style.width = `${Math.round((task.progress || 0) * 100)}%`;
    }

    if (stepsEl && task.steps && task.steps.length > 0) {
      stepsEl.innerHTML = task.steps.map((s) => {
        let cls = '';
        let dot = '<span class="dot dot-muted"></span>';
        if (s.status === 'done') { cls = 'done'; dot = '<span class="dot dot-success"></span>'; }
        if (s.status === 'active') { cls = 'active'; dot = '<span class="dot" style="background:var(--accent);animation:pulse 1.5s infinite"></span>'; }
        return `<li class="progress-step ${cls}">${dot} ${Utils.escapeHTML(s.name || s.label || s)}</li>`;
      }).join('');
    }

    // When task completes, mark card as done
    if (task.phase === 'completed' || task.phase === 'failed') {
      const fill = card.querySelector('.progress-bar-fill');
      if (fill) {
        fill.style.width = '100%';
        fill.style.background = task.phase === 'failed' ? 'var(--danger)' : 'var(--success)';
      }
      const title = card.querySelector('.progress-title');
      if (title) {
        title.textContent = (task.name || '任务') + (task.phase === 'failed' ? ' — 失败' : ' — 完成');
      }
      // Keep card for a moment then clean up
      setTimeout(() => {
        progressCards.delete(skillId);
      }, 10000);
    }

    scrollDown();
  }

  // ── Progress snapshot (static, embedded in agent reply) ──
  function renderProgressSnapshot(meta) {
    const card = document.createElement('div');
    card.className = 'embedded-card progress-card';
    const pct = Math.round((meta.progress || 0) * 100);
    card.innerHTML = `
      <div class="progress-title">${Utils.escapeHTML(meta.skill_name || '任务进度')}</div>
      <div class="progress-bar-track">
        <div class="progress-bar-fill" style="width:${pct}%"></div>
      </div>
    `;
    return card;
  }

  // ── Confirmation card ──
  function renderConfirmCard(conf) {
    const card = document.createElement('div');
    card.className = 'embedded-card confirm-card';
    card.innerHTML = `
      <div class="confirm-title">⚠️ ${Utils.escapeHTML(conf.title || '确认操作')}</div>
      <div class="confirm-detail">${Utils.escapeHTML(conf.detail || conf.message || '请确认此操作')}</div>
      <div class="confirm-actions">
        <button class="btn btn-secondary confirm-cancel-btn">取消</button>
        <button class="btn btn-primary confirm-approve-btn">确认执行</button>
      </div>
    `;

    const skillName = conf.skill_name || '';
    const skillArgs = conf.arguments || {};

    card.querySelector('.confirm-cancel-btn').addEventListener('click', () => {
      card.querySelector('.confirm-actions').innerHTML =
        '<span style="color:var(--text-disabled);font-size:var(--font-size-sm)">已取消</span>';
      WS.sendTurn('取消', { quick_action: 'cancel_skill', skill_name: skillName });
    });

    card.querySelector('.confirm-approve-btn').addEventListener('click', () => {
      card.querySelector('.confirm-actions').innerHTML =
        '<span style="color:var(--success);font-size:var(--font-size-sm)">已确认，执行中...</span>';
      WS.sendTurn('确认执行', { quick_action: 'approve_skill', skill_name: skillName, arguments: skillArgs });
    });

    return card;
  }

  // ── Recovery card ──
  function injectRecoveryCard(errorMsg) {
    // Deduplicate: only one recovery card at a time
    if (document.querySelector('.recovery-card')) return;

    const div = document.createElement('div');
    div.className = 'msg fade-in';
    div.innerHTML = `
      <div class="msg-avatar agent">⚠️</div>
      <div class="msg-body">
        <div class="embedded-card recovery-card">
          <div class="recovery-title">执行异常</div>
          <div class="recovery-detail">${Utils.escapeHTML(errorMsg || '机器人任务执行遇到问题')}</div>
          <div class="recovery-hint">建议：重新观察环境后再试，或检查机器人状态。</div>
          <div class="recovery-actions">
            <button class="btn btn-secondary recovery-stop-btn">停止任务</button>
            <button class="btn btn-secondary recovery-skip-btn">跳过</button>
            <button class="btn btn-primary recovery-retry-btn">重试</button>
          </div>
        </div>
      </div>
    `;

    div.querySelector('.recovery-retry-btn').addEventListener('click', () => {
      div.querySelector('.recovery-actions').innerHTML =
        '<span style="color:var(--accent);font-size:var(--font-size-sm)">正在重试...</span>';
      WS.sendTurn('重试', { quick_action: 'retry_skill' });
    });

    div.querySelector('.recovery-skip-btn').addEventListener('click', () => {
      div.querySelector('.recovery-actions').innerHTML =
        '<span style="color:var(--text-disabled);font-size:var(--font-size-sm)">已跳过</span>';
      WS.sendTurn('跳过', { quick_action: 'skip_skill' });
      Store.set('recoveryRequired', false);
    });

    div.querySelector('.recovery-stop-btn').addEventListener('click', () => {
      div.querySelector('.recovery-actions').innerHTML =
        '<span style="color:var(--danger);font-size:var(--font-size-sm)">正在停止...</span>';
      WS.sendTurn('停止任务', { quick_action: 'stop_task' });
      Store.set('recoveryRequired', false);
    });

    messagesInner.appendChild(div);
    scrollDown();
  }

  // ── Thinking indicator ──
  function showThinking() {
    if (thinkingMsgEl) {
      thinkingMsgEl.remove();
      thinkingMsgEl = null;
    }
    const div = document.createElement('div');
    div.className = 'msg msg-thinking fade-in';
    div.innerHTML = `
      <div class="msg-avatar agent">🤖</div>
      <div class="msg-body">
        <div class="msg-thinking">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    messagesInner.appendChild(div);
    thinkingMsgEl = div;
    scrollDown();
  }

  function hideThinking() {
    if (thinkingMsgEl) {
      thinkingMsgEl.remove();
      thinkingMsgEl = null;
    }
  }

  // ── Lightbox ──
  function openLightbox(images, startIdx) {
    const lb = $('#lightbox');
    const img = $('#lightbox-img');
    const prev = $('#lightbox-prev');
    const next = $('#lightbox-next');

    if (!images || images.length === 0) return;

    let currentIdx = startIdx || 0;

    function show(i) {
      currentIdx = i;
      img.src = images[i];
      prev.style.display = i > 0 ? '' : 'none';
      next.style.display = i < images.length - 1 ? '' : 'none';
    }

    prev.onclick = (ev) => { ev.stopPropagation(); show(currentIdx - 1); };
    next.onclick = (ev) => { ev.stopPropagation(); show(currentIdx + 1); };

    show(currentIdx);
    lb.style.display = 'flex';

    document.addEventListener('keydown', function escHandler(ev) {
      if (ev.key === 'Escape') {
        lb.style.display = 'none';
        document.removeEventListener('keydown', escHandler);
      }
    });
  }

  // ── Estop ──
  function showEstopConfirm() {
    $('#confirm-dialog').style.display = 'flex';
  }
  async function confirmEstop() {
    $('#confirm-dialog').style.display = 'none';
    try {
      await API.sendTurn('紧急停止', { quick_action: 'stop_motion' });
      showToast('急停指令已发送');
    } catch (e) {
      showToast('急停发送失败: ' + e.message, true);
    }
  }

  // ── Panel toggles ──
  function toggleSidebar(force) {
    const sidebar = $('#sidebar');
    const show = typeof force === 'boolean' ? force : sidebar.classList.contains('collapsed');
    if (show) {
      sidebar.classList.remove('collapsed');
      $('#sidebar-expand').style.display = 'none';
    } else {
      sidebar.classList.add('collapsed');
      $('#sidebar-expand').style.display = '';
    }
  }

  function toggleRightPanel(force) {
    const panel = $('#right-panel');
    const show = typeof force === 'boolean' ? force : panel.classList.contains('collapsed');
    if (show) {
      panel.classList.remove('collapsed');
      $('#right-panel-expand').style.display = 'none';
    } else {
      panel.classList.add('collapsed');
      $('#right-panel-expand').style.display = '';
    }
  }

  function toggleMobilePanel(force) {
    const panel = $('#mobile-panel');
    const show = typeof force === 'boolean' ? force : panel.style.display === 'none';
    panel.style.display = show ? 'flex' : 'none';
  }

  // ── History sidebar ──
  function updateHistorySidebar(conv) {
    if (!historyList) return;
    const userTurns = conv.filter(m => m.role === 'user');
    const latest = userTurns[userTurns.length - 1];
    const preview = latest ? Utils.truncate(latest.content, 24) : '暂无消息';
    historyList.innerHTML = `<div class="sidebar-item conversation-current active" data-episode="current">
      <span>当前对话</span>
      <small title="${Utils.escapeHTML(preview)}">${Utils.escapeHTML(preview)}</small>
    </div>`;
  }

  // ── Helpers ──
  function scrollDown() {
    requestAnimationFrame(() => {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  }

  function showToast(text, isError) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const div = document.createElement('div');
    div.className = 'toast' + (isError ? ' error' : '');
    div.textContent = text;
    document.body.appendChild(div);

    setTimeout(() => div.remove(), 3000);
  }

})();
