/* Hey Robot — Admin Panel Logic */

(function () {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  async function init() {
    await Promise.all([
      loadConfig(),
      loadHealth(),
      loadRuntimeSummary(),
    ]);
  }

  async function loadConfig() {
    try {
      const cfg = await API.loadConfig();
      $('#dep-id').textContent = cfg.deployment_id || '--';
      $('#dep-channel').textContent = cfg.channel || '--';
      $('#dep-host').textContent = accessHost(cfg.access_url, cfg.listen_host);
      $('#dep-port').textContent = cfg.listen_port || '--';
      const features = cfg.features || {};
      const enabled = Object.entries(features)
        .filter(([, v]) => v)
        .map(([key]) => featureLabel(key));
      $('#health-features').innerHTML = enabled.length > 0
        ? enabled.map(label => `<span class="feature-chip">${Utils.escapeHTML(label)}</span>`).join('')
        : '--';
    } catch (e) {
      console.debug('Config load failed', e.message);
    }
  }

  async function loadHealth() {
    try {
      const resp = await fetch('/health');
      const data = await resp.json();
      const badge = $('#health-badge');
      if (data.status === 'ok') {
        badge.className = 'badge badge-ok';
        badge.textContent = '正常';
      } else {
        badge.className = 'badge badge-warn';
        badge.textContent = data.status || '异常';
      }
    } catch (e) {
      const badge = $('#health-badge');
      badge.className = 'badge badge-err';
      badge.textContent = '无法连接';
    }
  }

  async function loadRuntimeSummary() {
    try {
      const data = await API.get('/api/runtime-summary?limit=50');
      renderStats(data.stats || {});
      renderTasks(data.tasks || []);
      renderRobots(data.robots || []);
      renderEvents(data.events || []);
      renderSkills(data.skills || []);
    } catch (e) {
      console.debug('Runtime summary failed, using legacy endpoints', e.message);
      await loadEventsLegacy();
      await loadRepliesLegacy();
    }
  }

  function renderStats(stats) {
    $('#stat-tasks').textContent = stats.task_count || 0;
    $('#stat-robots').textContent = stats.robot_count || 0;
    $('#stat-skills').textContent = stats.skill_count || 0;
    $('#stat-events').textContent = stats.event_count || 0;
  }

  function renderTasks(tasks) {
    const list = $('#tasks-ops-list');
    if (tasks.length === 0) {
      list.innerHTML = '<div class="empty-state">暂无任务</div>';
      return;
    }
    list.innerHTML = tasks.slice(0, 15).map((t) => {
      const badgeCls = statusBadgeCls(t.status);
      const badgeLabel = statusLabel(t.status);
      const ts = t.updated_at ? Utils.formatDate(t.updated_at * 1000) + ' ' + Utils.formatTime(t.updated_at * 1000) : '';
      return `<div class="event-row">
        <div style="display:flex;align-items:center;gap:8px">
          <span class="badge ${badgeCls}">${badgeLabel}</span>
          <span class="summary">${Utils.escapeHTML(Utils.truncate(t.root_task || t.task_id || '(未命名)', 60))}</span>
        </div>
        <div class="meta">${Utils.escapeHTML(t.robot_id || '')} · ${ts}</div>
      </div>`;
    }).join('');
  }

  function renderRobots(robots) {
    const list = $('#robots-ops-list');
    if (!robots || robots.length === 0) {
      list.innerHTML = '<div class="empty-state">暂无机器人状态</div>';
      return;
    }
    list.innerHTML = robots.slice(0, 5).map((r) => {
      const state = typeof r.state === 'string'
        ? r.state
        : (r.state && typeof r.state === 'object' ? r.state.state : 'unknown');
      const status = r.status && typeof r.status === 'object' ? r.status : {};
      const stateLabel = Utils.robotStateLabel(state || 'unknown');
      const ts = r.updated_at ? Utils.formatTime(r.updated_at * 1000) : '';
      const frameText = status.frame_id != null ? `帧 ${status.frame_id}` : '';
      const errorText = status.error ? ` · ${status.error}` : '';
      let activeTaskText = '';
      if (typeof r.active_task === 'string') activeTaskText = r.active_task;
      else if (r.active_task && typeof r.active_task === 'object') activeTaskText = r.active_task.root_task || r.active_task.task || r.active_task.summary || '';
      return `<div class="event-row">
        <div style="display:flex;align-items:center;gap:8px">
          <span class="dot ${state === 'idle' || state === 'executing' ? 'dot-success' : status.error ? 'dot-danger' : 'dot-muted'}"></span>
          <span class="v">${Utils.escapeHTML(r.robot_id || '--')}</span>
          <span style="color:var(--text-secondary);font-size:var(--font-size-xs)">${stateLabel}</span>
        </div>
        <div class="summary">${Utils.escapeHTML(activeTaskText || (state === 'idle' ? '无活动任务' : '--'))}</div>
        <div class="meta">${Utils.escapeHTML(frameText + errorText)}${frameText || errorText ? ' · ' : ''}${ts}</div>
      </div>`;
    }).join('');
  }

  function renderEvents(events) {
    const list = $('#events-list');
    if (events.length === 0) {
      list.innerHTML = '<div class="empty-state">暂无事件</div>';
      return;
    }
    list.innerHTML = events.slice(0, 15).map((ev) => {
      const kind = ev.kind || '';
      const summary = ev.summary || ev.text || '';
      const ts = ev.timestamp ? Utils.formatDate(ev.timestamp * 1000) + ' ' + Utils.formatTime(ev.timestamp * 1000) : '';
      return `<div class="event-row">
        <div class="kind">${Utils.escapeHTML(kind)}</div>
        ${summary ? `<div class="summary">${Utils.escapeHTML(Utils.truncate(summary, 120))}</div>` : ''}
        <div class="meta">${ts}</div>
      </div>`;
    }).join('');
  }

  function renderSkills(skills) {
    const list = $('#skills-ops-list');
    if (!skills || skills.length === 0) {
      list.innerHTML = '<div class="empty-state">暂无技能记录</div>';
      return;
    }
    const ordered = skills.slice().sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
    list.innerHTML = ordered.slice(0, 12).map((s, index) => {
      const rawName = s.name || s.skill || s.skill_id || '';
      const name = skillLabel(rawName);
      const phase = s.phase || '';
      const badgeCls = phase === 'completed' ? 'badge-completed' : phase === 'failed' ? 'badge-failed' : 'badge-pending';
      const phaseLabel = phase === 'completed' ? '已完成' : phase === 'failed' ? '失败' : phase === 'executing' ? '执行中' : phase || '未知';
      const detail = s.error || s.summary || s.objective || '';
      const ts = s.updated_at
        ? Utils.formatDate(s.updated_at * 1000) + ' ' + Utils.formatTime(s.updated_at * 1000)
        : '';
      return `<div class="event-row">
        <div style="display:flex;align-items:center;gap:8px">
          ${index === 0 ? '<span class="badge badge-active">最新</span>' : ''}
          <span class="badge ${badgeCls}">${Utils.escapeHTML(phaseLabel)}</span>
          <span class="summary" title="${Utils.escapeHTML(rawName)}">${Utils.escapeHTML(Utils.truncate(name, 40))}</span>
        </div>
        ${detail ? `<div class="summary">${Utils.escapeHTML(Utils.truncate(detail, 100))}</div>` : ''}
        <div class="meta">${Utils.escapeHTML(s.robot_id || '')}${s.robot_id && ts ? ' · ' : ''}${ts}</div>
      </div>`;
    }).join('');
  }

  async function loadEventsLegacy() {
    const list = $('#events-list');
    try {
      const data = await API.get('/events/recent?limit=20');
      const events = data.events || [];
      if (events.length === 0) {
        list.innerHTML = '<div class="empty-state">暂无事件</div>';
        return;
      }
      list.innerHTML = events.slice().reverse().map((ev) => {
        const payload = ev.payload || {};
        const kind = ev.kind || ev.type || payload.kind || '';
        const summary = payload.summary || payload.text || payload.error || '';
        const ts = ev.timestamp ? Utils.formatDate(ev.timestamp * 1000) + ' ' + Utils.formatTime(ev.timestamp * 1000) : '';
        return `<div class="event-row">
          <div class="kind">${Utils.escapeHTML(kind)}</div>
          ${summary ? `<div class="summary">${Utils.escapeHTML(Utils.truncate(summary, 120))}</div>` : ''}
          <div class="meta">${ts}</div>
        </div>`;
      }).join('');
    } catch (e) {
      list.innerHTML = '<div class="empty-state">加载失败: ' + Utils.escapeHTML(e.message) + '</div>';
    }
  }

  async function loadRepliesLegacy() {
    const list = $('#tasks-ops-list');
    try {
      const data = await API.get('/replies/recent?limit=20');
      const replies = data.replies || [];
      if (replies.length === 0) {
        list.innerHTML = '<div class="empty-state">暂无回复</div>';
        return;
      }
      list.innerHTML = replies.slice().reverse().map((r) => {
        const payload = r.payload || r;
        const text = payload.text || '';
        const ts = payload.timestamp ? Utils.formatDate(payload.timestamp * 1000) + ' ' + Utils.formatTime(payload.timestamp * 1000) : '';
        return `<div class="event-row">
          <div class="summary">${Utils.escapeHTML(Utils.truncate(text, 200))}</div>
          <div class="meta">${ts}</div>
        </div>`;
      }).join('');
    } catch (e) {
      list.innerHTML = '<div class="empty-state">加载失败: ' + Utils.escapeHTML(e.message) + '</div>';
    }
  }

  function statusBadgeCls(status) {
    const map = { active: 'badge-active', executing: 'badge-active',
      completed: 'badge-completed', failed: 'badge-failed',
      cancelled: 'badge-cancelled', paused: 'badge-cancelled' };
    return map[status] || 'badge-pending';
  }

  function statusLabel(status) {
    const map = { active: '进行中', executing: '执行中',
      completed: '已完成', failed: '失败', cancelled: '已取消', paused: '已暂停' };
    return map[status] || status || '--';
  }

  function accessHost(accessUrl, listenHost) {
    try {
      return accessUrl ? new URL(accessUrl).hostname : (listenHost || '--');
    } catch (_e) {
      return listenHost || '--';
    }
  }

  function featureLabel(key) {
    const labels = {
      history: '对话历史',
      identity_binding: '账号绑定',
      recent_replies: '最近回复',
      runtime_events: '运行事件',
      websocket: '实时连接',
      execution_feedback: '执行反馈',
      cockpit: '任务监控',
    };
    return labels[key] || key;
  }

  function skillLabel(name) {
    const labels = {
      inspect_scene: '观察场景',
      look_around: '环顾四周',
      detect_marker: '识别标记',
      move_base: '底盘移动',
      turn_base: '底盘转向',
      set_gripper: '控制夹爪',
      set_arm_pose: '设置机械臂姿态',
      move_arm_joints: '移动机械臂关节',
      human_follow: '跟随人员',
      stop_motion: '停止运动',
    };
    return labels[name] || name || '未知技能';
  }

  init();
})();
