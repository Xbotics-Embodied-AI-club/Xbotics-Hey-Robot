/* Hey Robot — HTTP API helpers */

const API = (() => {
  async function request(method, path, body) {
    const opts = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);

    const resp = await fetch(path, opts);
    if (!resp.ok) {
      const text = await resp.text().catch(() => '');
      throw new Error(text || `${resp.status} ${resp.statusText}`);
    }
    return resp.json();
  }

  function get(path) { return request('GET', path); }
  function post(path, body) { return request('POST', path, body); }

  async function sendTurn(text, metadata) {
    const payload = {
      text,
      metadata: metadata || {},
      chat_id: localStorage.getItem('chat_id') || 'web',
      sender_id: localStorage.getItem('sender_id') || 'web-user',
    };
    return post('/turn', payload);
  }

  async function loadHistory(limit) {
    return get(`/history?limit=${limit || 50}`);
  }

  async function loadCockpit(episodeId) {
    return get(`/cockpit/${episodeId}`);
  }

  async function loadConfig() {
    return get('/config.json');
  }

  return { get, post, sendTurn, loadHistory, loadCockpit, loadConfig };
})();
