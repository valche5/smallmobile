import type { ClientMsg, ServerMsg } from '../shared/protocol';

export type NetStatus = 'connecting' | 'open' | 'closed';

interface Handlers {
  message(msg: ServerMsg): void;
  status(s: NetStatus): void;
  /** Appelé à chaque (re)connexion : c'est là qu'on s'identifie auprès du serveur. */
  open(): void;
}

/** Connexion WebSocket au serveur, qui se rétablit toute seule (téléphone en veille, Wi-Fi capricieux…). */
export class Connection {
  private ws: WebSocket | null = null;
  private retry = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private stopped = true;

  constructor(private handlers: Handlers) {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && !this.stopped && !this.ws) this.connect();
    });
    window.addEventListener('online', () => {
      if (!this.stopped && !this.ws) this.connect();
    });
  }

  connect() {
    this.stopped = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.ws) return;
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    this.ws = ws;
    this.handlers.status('connecting');
    ws.onopen = () => {
      this.retry = 0;
      this.handlers.status('open');
      this.handlers.open();
    };
    ws.onmessage = (e) => {
      try {
        this.handlers.message(JSON.parse(String(e.data)) as ServerMsg);
      } catch (err) {
        console.error('Message du serveur illisible', err);
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.handlers.status('closed');
      if (!this.stopped) {
        const delay = Math.min(8000, 500 * 2 ** this.retry++);
        this.timer = setTimeout(() => this.connect(), delay);
      }
    };
    ws.onerror = () => ws.close();
  }

  send(msg: ClientMsg): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  close() {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    ws?.close();
    this.handlers.status('closed');
  }
}
