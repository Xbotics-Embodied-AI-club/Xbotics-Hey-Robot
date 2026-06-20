/* Hey Robot — utility helpers */

const Utils = (() => {
  function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function formatTime(ts) {
    const d = new Date(ts);
    const h = String(d.getHours()).padStart(2, '0');
    const m = String(d.getMinutes()).padStart(2, '0');
    return `${h}:${m}`;
  }

  function formatDate(ts) {
    const d = new Date(ts);
    const y = d.getFullYear();
    const mo = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${mo}-${day}`;
  }

  function batteryColor(pct) {
    if (pct == null) return 'var(--text-disabled)';
    if (pct > 50) return 'var(--success)';
    if (pct > 20) return 'var(--warning)';
    return 'var(--danger)';
  }

  function batteryLabel(pct) {
    if (pct == null) return '--';
    return `${Math.round(pct)}%`;
  }

  function robotStateLabel(state) {
    const map = { idle: '空闲', executing: '执行中', skill_completed: '刚完成', failed: '异常', degraded: '降级', closed: '离线' };
    return map[state] || state || '未知';
  }

  function truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.slice(0, max) + '...' : str;
  }

  /** Debounce fn calls by `ms` milliseconds */
  function debounce(fn, ms) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }

  /** Load a script element and return a promise */
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const el = document.createElement('script');
      el.src = src;
      el.onload = resolve;
      el.onerror = reject;
      document.head.appendChild(el);
    });
  }

  return { escapeHTML, formatTime, formatDate, batteryColor, batteryLabel, robotStateLabel, truncate, debounce, loadScript };
})();
