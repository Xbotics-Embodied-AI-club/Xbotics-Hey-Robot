export function statusLevel(value) {
  const text = String(value || "").toLowerCase();
  if (["error", "failed", "failure", "aborted", "timeout", "critical", "fatal"].some((token) => text.includes(token))) {
    return "bad";
  }
  if (["warn", "warning", "recover", "retry", "blocked", "unknown", "degraded", "disabled", "stale"].some((token) => text.includes(token))) {
    return "warn";
  }
  return "ok";
}

export function skillLevel(skill) {
  return statusLevel(skill?.phase || skill?.status || skill?.error);
}

export function eventLevel(event) {
  return statusLevel(event?.severity || event?.kind || event?.payload?.error);
}
