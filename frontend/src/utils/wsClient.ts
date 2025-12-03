let ws: WebSocket | null = null;

type MessageHandler = (msg: string) => void;
type StatusHandler = () => void;

let onMessageCb: MessageHandler | null = null;
let onOpenCb: StatusHandler | null = null;
let onCloseCb: StatusHandler | null = null;

let shouldReconnect = false;
let reconnectTimer: number | null = null;
let lastUrl: string | null = null;
const RECONNECT_DELAY_MS = 3000;

export function connectWebSocket(
  url: string,
  handlers: {
    onMessage?: MessageHandler;
    onOpen?: StatusHandler;
    onClose?: StatusHandler;
  } = {},
  autoReconnect: boolean = true
) {
  // Remember for future reconnects
  lastUrl = url;
  shouldReconnect = autoReconnect;
  onMessageCb = handlers.onMessage ?? null;
  onOpenCb = handlers.onOpen ?? null;
  onCloseCb = handlers.onClose ?? null;

  // Avoid opening multiple connections
  if (
    ws &&
    (ws.readyState === WebSocket.OPEN ||
      ws.readyState === WebSocket.CONNECTING)
  ) {
    return;
  }

  if (reconnectTimer != null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("[WS] connected");
    onOpenCb && onOpenCb();
  };

  ws.onmessage = (event) => {
    const text = String(event.data);
    onMessageCb && onMessageCb(text);
  };

  ws.onclose = () => {
    console.log("[WS] closed");
    ws = null;
    onCloseCb && onCloseCb();

    if (shouldReconnect && lastUrl) {
      console.log("[WS] scheduling reconnect in", RECONNECT_DELAY_MS, "ms");
      reconnectTimer = window.setTimeout(() => {
        connectWebSocket(lastUrl!, handlers, true);
      }, RECONNECT_DELAY_MS);
    }
  };

  ws.onerror = (err) => {
    console.error("[WS] error", err);
  };
}

export function sendWsMessage(text: string) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.warn("[WS] not open – cannot send");
    return;
  }
  ws.send(text);
}

export function disableAutoReconnect() {
  shouldReconnect = false;
  if (reconnectTimer != null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

export function closeWebSocket() {
  if (ws) {
    ws.close();
    ws = null;
  }
}