import type { Server, ServerWebSocket, WebSocketHandler } from 'bun';
import type { ClientMsg } from '../shared/protocol';
import type { Client, RoomManager } from './rooms';

export interface SocketData {
  client: Client;
}

/** Transforme une requête HTTP sur /ws en connexion WebSocket. */
export function upgradeSocket(req: Request, srv: Server<SocketData>): Response | undefined {
  const client: Client = { code: null, seat: null, send: () => {} };
  if (srv.upgrade(req, { data: { client } })) return undefined;
  return new Response('Connexion WebSocket attendue', { status: 400 });
}

/** Branche les WebSockets sur le gestionnaire de parties. */
export function socketHandlers(rooms: RoomManager, sockets: Set<ServerWebSocket<SocketData>>): WebSocketHandler<SocketData> {
  return {
    perMessageDeflate: true,
    idleTimeout: 60,
    maxPayloadLength: 64 * 1024,
    open(ws) {
      sockets.add(ws);
      ws.data.client.send = (msg) => {
        ws.send(JSON.stringify(msg));
      };
    },
    message(ws, raw) {
      let msg: ClientMsg;
      try {
        msg = JSON.parse(String(raw)) as ClientMsg;
      } catch {
        return;
      }
      rooms.handle(ws.data.client, msg);
    },
    close(ws) {
      sockets.delete(ws);
      rooms.disconnect(ws.data.client);
    },
  };
}
