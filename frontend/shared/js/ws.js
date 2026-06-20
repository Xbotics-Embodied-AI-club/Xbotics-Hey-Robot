/* Hey Robot — WebSocket connection manager */

const WS = (() => {
  let _socket = null;
  let _url = null;
  let _reconnectTimer = null;
  let _reconnectAttempts = 0;
  const _maxReconnect = 10;

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    _url = `${proto}://${location.host}/ws`;

    _socket = new WebSocket(_url);

    _socket.onopen = () => {
      Store.set('wsConnected', true);
      _reconnectAttempts = 0;
      console.debug('WS connected');
    };

    _socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.debug('WS recv', data.type, data.payload?.text?.slice(0, 30) || '');
        handleMessage(data);
      } catch (e) {
        console.debug('WS unparseable message', e);
      }
    };

    _socket.onclose = () => {
      Store.set('wsConnected', false);
      _socket = null;
      scheduleReconnect();
    };

    _socket.onerror = () => {
      // onclose will fire after this
    };
  }

  function scheduleReconnect() {
    if (_reconnectTimer) return;
    if (_reconnectAttempts >= _maxReconnect) return;
    _reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, _reconnectAttempts), 30000);
    console.debug(`WS reconnect #${_reconnectAttempts} in ${delay}ms`);
    _reconnectTimer = setTimeout(() => {
      _reconnectTimer = null;
      connect();
    }, delay);
  }

  function handleMessage(data) {
    const type = data.type;

    if (type === 'agent.reply') {
      const payload = data.payload || {};
      const text = payload.text || '';
      const metadata = payload.metadata || {};
      // Include media (images) from the reply payload itself
      if (payload.media && payload.media.length > 0) {
        metadata.images = payload.media;
      }
      if (text) {
        addMessage({ role: 'agent', content: text, timestamp: Date.now(), metadata });
      }
    } else if (type === 'runtime.event') {
      handleRuntimeEvent(data.payload || {});
    }
  }

  function handleRuntimeEvent(payload) {
    const kind = payload.kind || payload.event || '';

    // Robot status update
    if (kind === 'robot.status' || kind === 'ROBOT_STATUS') {
      const data = payload.payload || payload;
      updateRobotState(data);
    }

    // Robot observation (camera images)
    if (kind === 'robot.observation' || kind === 'ROBOT_OBSERVATION') {
      const data = payload.payload || payload;
      handleRobotObservation(data);
    }

    // Skill lifecycle
    if (kind === 'skill.lifecycle' || kind === 'SKILL_LIFECYCLE') {
      const data = payload.payload || payload;
      handleSkillLifecycle(data);
    }
  }

  function handleRobotObservation(data) {
    const images = (data.images || []).map((img) => ({
      url: img.url || img.uri || '',
      uri: img.uri || img.url || '',
      camera: img.camera || '',
      width: img.width,
      height: img.height,
    }));

    if (images.length > 0) {
      addMessage({
        role: 'agent',
        content: data.task || '观测画面',
        timestamp: Date.now(),
        metadata: { images, frame_id: data.frame_id },
      });
    }
  }

  function updateRobotState(data) {
    const state = data.state || '';
    const metrics = data.metrics || {};
    const battery = metrics.battery || {};
    const readiness = metrics.readiness || {};

    Store.update('robot', {
      online: state !== 'closed' && state !== '',
      battery: battery.percentage || battery.status || null,
      batteryPct: typeof battery.percentage === 'number' ? battery.percentage : null,
      base: readiness.base?.ok ? '就绪' : (state === 'degraded' ? '异常' : '未知'),
      arm: readiness.arm?.ok ? '就绪' : (state === 'degraded' ? '异常' : '未知'),
      camera: readiness.camera?.ok ? '正常' : (state === 'degraded' ? '异常' : '未知'),
      state,
      readiness,
      error: data.error || null,
    });

    // Recovery banner
    if (data.error && state === 'failed') {
      Store.set('recoveryRequired', true);
    }

    Store.set('loading', false);
  }

  function handleSkillLifecycle(data) {
    const phase = data.phase || '';
    const skillId = data.skill_id || '';

    if (phase === 'executing' || phase === 'accepted') {
      const steps = data.steps || buildDefaultSteps(data);
      const stepsExecuted = typeof data.steps_executed === 'number' ? data.steps_executed : null;
      const progress = typeof data.progress === 'number'
        ? data.progress
        : (stepsExecuted != null && steps.length > 0
          ? Math.min(stepsExecuted / steps.length, 1)
          : 0);
      Store.update('task', {
        active: true,
        skillId,
        name: data.name || data.skill || '',
        phase,
        summary: data.summary || '',
        progress,
        steps,
      });
    } else if (phase === 'completed') {
      Store.update('task', {
        active: false,
        skillId,
        phase: 'completed',
        summary: data.summary || '任务完成',
        progress: 1,
        steps: data.steps || [],
      });
    } else if (phase === 'failed') {
      Store.update('task', {
        active: false,
        skillId,
        phase: 'failed',
        summary: data.error || data.summary || '任务失败',
        progress: data.progress || 0,
        steps: data.steps || [],
      });
      Store.set('recoveryRequired', true);
    }
  }

  function buildDefaultSteps(data) {
    const steps = [];
    if (data.name) {
      steps.push({ name: data.name, status: 'active' });
    }
    return steps;
  }

  function addMessage(msg) {
    const conv = Store.get('conversation');
    const updated = [...conv, msg];
    Store.set('conversation', updated);
  }

  function sendTurn(text, metadata) {
    if (!_socket || _socket.readyState !== WebSocket.OPEN) {
      console.debug('WS not open, using HTTP fallback for turn');
      return API.sendTurn(text, metadata).then(r => ({ ...r, _httpFallback: true }));
    }
    console.debug('WS sending turn, socket state=', _socket.readyState);

    const payload = {
      text,
      metadata: metadata || {},
      chat_id: getChatId(),
      sender_id: getSenderId(),
    };

    _socket.send(JSON.stringify(payload));
    return Promise.resolve({ accepted: true });
  }

  /** Poll /replies/recent for a reply matching the given trace_id. */
  async function pollReply(traceId, timeoutMs = 30000) {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      try {
        const resp = await fetch(`/replies/recent?trace_id=${encodeURIComponent(traceId)}&limit=1`);
        if (resp.ok) {
          const data = await resp.json();
          const replies = data.replies || [];
          if (replies.length > 0) {
            const r = replies[0];
            const payload = r.payload || r;
            const text = payload.text || r.text || '';
            if (text) {
              addMessage({ role: 'agent', content: text, timestamp: Date.now(), metadata: payload.metadata || {} });
              return true;
            }
          }
        }
      } catch (e) { /* retry */ }
      await new Promise(r => setTimeout(r, 800));
    }
    return false;
  }

  function isConnected() {
    return _socket !== null && _socket.readyState === WebSocket.OPEN;
  }

  function getChatId() {
    return localStorage.getItem('chat_id') || 'web';
  }
  function getSenderId() {
    return localStorage.getItem('sender_id') || 'web-user';
  }

  function close() {
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }
    if (_socket) {
      _socket.close();
      _socket = null;
    }
  }

  return { connect, close, sendTurn, addMessage, pollReply, isConnected };
})();
