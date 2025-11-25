const DB_NAME = "rag-chat-db";
const DB_VERSION = 1;
const STORE = "user_requests";
const MAX_ITEMS = 50;

export type UserRequestRecord = {
  id?: number;      // auto-increment key
  text: string;
  ts: number;       // timestamp
};

function normalize(s: string): string {
  return s
    .trim()              // remove spaces at start/end
    .replace(/\s+/g, " ") // collapse multiple spaces
    .toLowerCase();       // unify case
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        store.createIndex("ts", "ts", { unique: false });
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function addUserRequest(text: string): Promise<void> {
  const normalized = normalize(text);
  if (!normalized) return
    const db = await openDB();

// Load existing history
  const all = await new Promise<UserRequestRecord[]>((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const store = tx.objectStore(STORE);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as UserRequestRecord[]);
    req.onerror = () => reject(req.error);
  });

  // Find duplicates (same normalized text)
  const duplicates = all.filter(r => normalize(r.text) === normalized);

  // Remove duplicates first
  if (duplicates.length > 0) {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      const store = tx.objectStore(STORE);
      duplicates.forEach(d => d.id != null && store.delete(d.id));
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  // Add as most recent
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    store.add({ text, ts: Date.now() } satisfies UserRequestRecord);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });

  // Trim to last MAX_ITEMS
  await trimRequests();
}

export async function getLastUserRequests(limit = MAX_ITEMS): Promise<UserRequestRecord[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const store = tx.objectStore(STORE);

    // Read all, then take last N (simple & reliable for small MAX)
    const req = store.getAll();

    req.onsuccess = () => {
      const all = (req.result as UserRequestRecord[]) || [];
      const last = all.slice(-limit);
      resolve(last);
    };
    req.onerror = () => reject(req.error);
  });
}

export async function clearUserRequests(): Promise<void> {
  const db = await openDB();

  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);

    store.clear();

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function trimRequests(): Promise<void> {
  const db = await openDB();

  const all = await new Promise<UserRequestRecord[]>((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const store = tx.objectStore(STORE);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as UserRequestRecord[]);
    req.onerror = () => reject(req.error);
  });

  if (all.length <= MAX_ITEMS) return;

  const toDelete = all.slice(0, all.length - MAX_ITEMS);

  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);

    toDelete.forEach(r => {
      if (r.id != null) store.delete(r.id);
    });

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}