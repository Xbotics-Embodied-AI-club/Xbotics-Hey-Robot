/* Hey Robot — Task Views Logic */

(function () {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  // Detect page type from URL path
  const path = location.pathname;

  if (path === '/tasks') {
    initTaskList();
  } else if (path.startsWith('/tasks/')) {
    const episodeId = path.split('/tasks/')[1];
    initTaskDetail(episodeId);
  }

  async function initTaskList() {
    const listEl = $('#task-list');
    try {
      const data = await API.get('/api/tasks?limit=50');
      const tasks = data.tasks || [];
      if (tasks.length === 0) {
        listEl.innerHTML = '<div class="empty-state">暂无任务记录</div>';
        return;
      }
      listEl.innerHTML = tasks.map((t) => {
        const badgeCls = statusBadge(t.status);
        const badgeLabel = statusLabel(t.status);
        const created = t.created_at ? Utils.formatDate(t.created_at * 1000) + ' ' + Utils.formatTime(t.created_at * 1000) : '';
        const episodeLink = t.episode_id ? `/tasks/${Utils.escapeHTML(t.episode_id)}` : '#';
        return `<div class="task-card">
          <div class="task-card-header">
            <span class="task-card-title">
              <a href="${episodeLink}">${Utils.escapeHTML(t.root_task || t.task_id || '(未命名)')}</a>
            </span>
            <span class="badge ${badgeCls}">${badgeLabel}</span>
          </div>
          <div class="task-card-meta">
            <span>Episode: ${Utils.escapeHTML(Utils.truncate(t.episode_id || '', 16))}</span>
            <span>Robot: ${Utils.escapeHTML(t.robot_id || '--')}</span>
            <span>重试: ${t.retry_count || 0}</span>
            ${created ? `<span>${created}</span>` : ''}
          </div>
        </div>`;
      }).join('');
    } catch (e) {
      listEl.innerHTML = `<div class="empty-state">加载失败: ${Utils.escapeHTML(e.message)}</div>`;
    }
  }

  async function initTaskDetail(episodeId) {
    const contentEl = $('#detail-content');
    try {
      const data = await API.get(`/api/episodes/${encodeURIComponent(episodeId)}/task`);
      if (!data) {
        contentEl.innerHTML = '<div class="empty-state">任务不存在</div>';
        return;
      }
      const task = data.task || {};
      const robot = data.robot || {};

      const badgeCls = statusBadge(task.status);
      const badgeLabel = statusLabel(task.status);
      const created = task.created_at ? Utils.formatDate(task.created_at * 1000) + ' ' + Utils.formatTime(task.created_at * 1000) : '';

      let html = '';

      // Task overview
      html += `<div class="detail-card">
        <h3>任务概览</h3>
        <div class="kv-row"><span class="k">任务</span><span class="v">${Utils.escapeHTML(task.root_task || '(未命名)')}</span></div>
        <div class="kv-row"><span class="k">状态</span><span class="v"><span class="badge ${badgeCls}">${badgeLabel}</span></span></div>
        <div class="kv-row"><span class="k">Task ID</span><span class="v">${Utils.escapeHTML(task.task_id || '--')}</span></div>
        <div class="kv-row"><span class="k">Episode ID</span><span class="v">${Utils.escapeHTML(episodeId)}</span></div>
        <div class="kv-row"><span class="k">Robot</span><span class="v">${Utils.escapeHTML(task.robot_id || '--')}</span></div>
        <div class="kv-row"><span class="k">Agent</span><span class="v">${Utils.escapeHTML(task.agent_id || '--')}</span></div>
        ${created ? `<div class="kv-row"><span class="k">创建时间</span><span class="v">${created}</span></div>` : ''}
        ${task.retry_count ? `<div class="kv-row"><span class="k">重试次数</span><span class="v">${task.retry_count}</span></div>` : ''}
        ${task.failure_reason ? `<div class="kv-row"><span class="k">失败原因</span><span class="v" style="color:var(--danger)">${Utils.escapeHTML(task.failure_reason)}</span></div>` : ''}
      </div>`;

      // Robot state
      if (robot) {
        const state = robot.state || 'unknown';
        const stateLabel = Utils.robotStateLabel(state);
        html += `<div class="detail-card">
          <h3>机器人状态</h3>
          <div class="kv-row"><span class="k">状态</span><span class="v">${stateLabel}</span></div>
          <div class="kv-row"><span class="k">活跃任务</span><span class="v">${Utils.escapeHTML(robot.active_task || '--')}</span></div>
          ${robot.last_status ? `<div class="kv-row"><span class="k">最近状态</span><span class="v">${Utils.escapeHTML(String(robot.last_status))}</span></div>` : ''}
        </div>`;
      }

      // Attempts timeline
      const attempts = task.attempts || [];
      if (attempts.length > 0) {
        html += `<div class="detail-card">
          <h3>执行尝试 (${attempts.length})</h3>`;
        for (const a of attempts) {
          const aBadge = statusBadge(a.status);
          const aLabel = statusLabel(a.status);
          html += `<div class="attempt-item">
            <div class="attempt-text">${Utils.escapeHTML(a.text || a.objective || '(无描述)')} <span class="badge ${aBadge}" style="font-size:10px">${aLabel}</span></div>
            <div class="attempt-meta">
              ${a.skill_id ? `Skill: ${Utils.escapeHTML(Utils.truncate(a.skill_id, 20))}` : ''}
              ${a.skill ? ` · ${Utils.escapeHTML(a.skill)}` : ''}
            </div>
          </div>`;
        }
        html += '</div>';
      }

      contentEl.innerHTML = html;
      $('#detail-title').textContent = Utils.truncate(task.root_task || '任务详情', 40);

    } catch (e) {
      contentEl.innerHTML = `<div class="empty-state">加载失败: ${Utils.escapeHTML(e.message)}</div>`;
    }
  }

  function statusBadge(status) {
    const map = {
      active: 'badge-active', executing: 'badge-active',
      completed: 'badge-completed', success: 'badge-completed',
      failed: 'badge-failed', error: 'badge-failed',
      cancelled: 'badge-cancelled', paused: 'badge-cancelled',
    };
    return map[status] || 'badge-pending';
  }

  function statusLabel(status) {
    const map = {
      active: '进行中', executing: '执行中',
      completed: '已完成', success: '成功',
      failed: '失败', error: '异常',
      cancelled: '已取消', paused: '已暂停',
      pending: '等待中',
    };
    return map[status] || status || '未知';
  }
})();
