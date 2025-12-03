import { useEffect, useState } from "react";
import {
  addUserRequest,
  getLastUserRequests,
  clearUserRequests,
  UserRequestRecord,
} from "../utils/indexedDbHistory";

export function useUserHistory() {
  const [history, setHistory] = useState<UserRequestRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const items = await getLastUserRequests();
      setHistory(items);
      setLoading(false);
    })();
  }, []);

  const store = async (text: string) => {
    await addUserRequest(text);
    const items = await getLastUserRequests();
    setHistory(items);
  };

  const clear = async () => {
    await clearUserRequests();
    setHistory([]);
  };

  return { history, loading, store, clear };
}