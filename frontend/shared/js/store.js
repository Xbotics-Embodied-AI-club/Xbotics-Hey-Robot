/* Hey Robot — minimal pub/sub state store */

const Store = (() => {
  const _state = {
    robot: { online: false, battery: null, batteryPct: null, arm: 'unknown', camera: 'unknown' },
    task: null,
    conversation: [],
    wsConnected: false,
    recoveryRequired: false,
    loading: true,
  };
  const _listeners = {};

  function get(key) {
    return _state[key];
  }

  function set(key, value) {
    _state[key] = value;
    (listenersFor(key)).forEach(fn => fn(value));
  }

  function update(key, patch) {
    if (typeof patch === 'function') {
      _state[key] = patch(_state[key]);
    } else if (typeof _state[key] === 'object' && _state[key] !== null && !Array.isArray(_state[key])) {
      _state[key] = { ..._state[key], ...patch };
    } else {
      _state[key] = patch;
    }
    (listenersFor(key)).forEach(fn => fn(_state[key]));
  }

  function on(key, fn) {
    if (!_listeners[key]) _listeners[key] = [];
    _listeners[key].push(fn);
    return () => {
      _listeners[key] = _listeners[key].filter(f => f !== fn);
    };
  }

  function listenersFor(key) {
    return _listeners[key] || [];
  }

  return { get, set, update, on };
})();
