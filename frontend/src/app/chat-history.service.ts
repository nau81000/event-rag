import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class ChatHistoryService {
  private readonly DB_NAME = 'event-rag-chat';
  private readonly STORE_NAME = 'user_requests';
  private readonly DB_VERSION = 1;

  private dbPromise: Promise<IDBDatabase>;

  constructor() {
    this.dbPromise = this.openDb();
  }

  private openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;

        if (!db.objectStoreNames.contains(this.STORE_NAME)) {
          // ⚠ key still required but we do not use it
          db.createObjectStore(this.STORE_NAME, {
            autoIncrement: true,     // required for insert order
          });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /** Load last `limit` user requests in order */
  async loadLastRequests(limit = 50): Promise<string[]> {
    const db = await this.dbPromise;

    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.STORE_NAME, 'readonly');
      const store = tx.objectStore(this.STORE_NAME);

      // will return: [{text}, {text}, ...]
      const req = store.getAll();

      req.onsuccess = () => {
        const all = (req.result as { text: string }[]) || [];
        const last = all.slice(-limit);
        resolve(last.map(entry => entry.text));
      };

      req.onerror = () => reject(req.error);
    });
  }

  /** Save last `limit` user requests */
  async saveRequests(requests: string[], limit = 50): Promise<void> {
    const db = await this.dbPromise;

    // Extract last `limit` requests
    const last = requests.slice(-limit);

    // Remove duplicates – simplest possible
    const unique = Array.from(new Set(last.map(r => r.trim())));

    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.STORE_NAME, 'readwrite');
      const store = tx.objectStore(this.STORE_NAME);

      store.clear().onsuccess = () => {
        unique.forEach(text => {
          if (text) {
            store.add({ text });
          }
        });
      };

      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
}