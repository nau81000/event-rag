import { Injectable, OnDestroy } from '@angular/core';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { BehaviorSubject } from 'rxjs';

export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error'
  | 'reconnecting';

@Injectable({ providedIn: 'root' })
export class ChatSocketService implements OnDestroy {
  // adjust URL if needed
  private WS_URL = 'ws://localhost:8000/ws';

  private socket$: WebSocketSubject<string> | null = null;

  private messagesSubject = new BehaviorSubject<string | null>(null);
  messages$ = this.messagesSubject.asObservable();

  private statusSubject = new BehaviorSubject<ConnectionStatus>('disconnected');
  status$ = this.statusSubject.asObservable();

  // retry config
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 10;
  private readonly baseDelay = 1000; // 1s, grows each attempt

  connect() {
    if (this.socket$ && !this.socket$.closed) {
      return; // already connected
    }

    this.statusSubject.next(
      this.reconnectAttempts === 0 ? 'connecting' : 'reconnecting'
    );
    console.log('[WS] connecting to', this.WS_URL);

    this.socket$ = webSocket<string>({
      url: this.WS_URL,

      // keep raw strings
      deserializer: (e) => e.data as string,

      openObserver: {
        next: () => {
          console.log('[WS] open');
          this.reconnectAttempts = 0;
          this.statusSubject.next('connected');
        },
      },

      closeObserver: {
        next: (event) => {
          console.log('[WS] closed', event.code, event.reason);

          // if the server closed cleanly, you can choose whether to retry
          // here we treat abnormal closes as errors → retry
          if (!event.wasClean) {
            this.handleError();
          } else {
            this.statusSubject.next('disconnected');
          }
        },
      },
    });

    this.socket$.subscribe({
      next: (msg) => {
        // raw JSON string from server
        this.messagesSubject.next(msg);
      },
      error: (err) => {
        console.error('[WS] error', err);
        this.handleError();
      },
      complete: () => {
        console.log('[WS] complete');
        // treat complete like a clean disconnect
        this.statusSubject.next('disconnected');
      },
    });
  }

  send(text: string) {
    if (!this.socket$ || this.socket$.closed) {
      console.warn('[WS] cannot send — socket not connected');
      return;
    }
    console.log('[WS] sending', text);
    this.socket$.next(text);
  }

  private handleError() {
    this.statusSubject.next('error');

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WS] max reconnect attempts reached, giving up');
      this.statusSubject.next('disconnected');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.baseDelay * this.reconnectAttempts;
    console.warn(`[WS] reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  close() {
    this.socket$?.complete();
    this.socket$ = null;
    this.statusSubject.next('disconnected');
  }

  ngOnDestroy() {
    this.close();
  }
}