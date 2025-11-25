
import React, { useState, useEffect, useRef } from "react";
import { Flex, Text, Box, Input, HStack, IconButton } from "@chakra-ui/react";
import { useUserHistory } from "../hooks/useUserHistory";
import { useCityFromGeolocation } from "../utils/useCityFromGeolocation";
import { useTranslation } from "react-i18next";
import { ChatbotHeader } from "./ChatbotHeader.tsx";

type Role = "user" | "assistant";

type Message = {
  role: Role;
  content: string;
};

type ChatbotProps = {
  endpoint: string;
};

type EventPayload = {
  title_fr?: string;
  location_city?: string;
  location_department?: string;
  location_countrycode?: string;
  conditions_fr?: string;
  timings?: string;
};

type SearchResult = {
  payload: EventPayload;
}

type ChatResponse = {
  answer: SearchResult[];
};

export type Timing = {
  begin: string; // ISO string
  end: string;   // ISO string
};


/**
 * Transform a JSON string of timings into a list of grouped human-readable date ranges.
 *
 * Example input JSON:
 * [
 *   {"begin": "2025-10-13T09:00:00+02:00", "end": "2025-10-13T18:00:00+02:00"},
 *   ...
 * ]
 */
function groupTimings(timingsStr: string | null | undefined): string[] {
  if (!timingsStr) {
    return ["Dates non précisées"];
  }

  let timings: Timing[];
  try {
    timings = JSON.parse(timingsStr);
  } catch {
    return ["Dates non lisibles"];
  }

  if (!Array.isArray(timings) || timings.length === 0) {
    return ["Dates non précisées"];
  }

  // Parse ISO timestamps into Date objects
  let slots = timings.map(t => {
    const begin = new Date(t.begin);
    const end = new Date(t.end);
    return { begin, end };
  });

  // Sort by begin date
  slots.sort((a, b) => a.begin.getTime() - b.begin.getTime());

  const groups: {
    start: Date;
    end: Date;
    hb: string; // hour begin
    he: string; // hour end
  }[] = [];

  // Format hour as "HHhMM"
  const fmtHour = (d: Date) =>
    d.toLocaleTimeString("fr-FR", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).replace(":", "h");

  let currentStart = slots[0].begin;
  let currentEnd = slots[0].end;
  let currentBeginTime = fmtHour(currentStart);
  let currentEndTime = fmtHour(currentEnd);

  for (let i = 1; i < slots.length; i++) {
    const begin = slots[i].begin;
    const end = slots[i].end;

    const isNextDay =
      begin.getDate() === currentEnd.getDate() + 1 &&
      begin.getMonth() === currentEnd.getMonth() &&
      begin.getFullYear() === currentEnd.getFullYear();

    const sameTime =
      fmtHour(begin) === currentBeginTime &&
      fmtHour(end) === currentEndTime;

    if (isNextDay && sameTime) {
      // Extend group
      currentEnd = end;
    } else {
      // Close current group
      groups.push({
        start: currentStart,
        end: currentEnd,
        hb: currentBeginTime,
        he: currentEndTime,
      });

      currentStart = begin;
      currentEnd = end;
      currentBeginTime = fmtHour(begin);
      currentEndTime = fmtHour(end);
    }
  }

  // Push last group
  groups.push({
    start: currentStart,
    end: currentEnd,
    hb: currentBeginTime,
    he: currentEndTime,
  });

  // Format final human-readable output
  const humanGroups: string[] = [];

  const fmtDate = (d: Date) =>
    d.toLocaleDateString("fr-FR", {
      weekday: "long",
      day: "2-digit",
      month: "long",
      year: "numeric",
    });

  groups.forEach((g) => {
    const startStr = fmtDate(g.start);
    const endStr = fmtDate(g.end);

    if (startStr === endStr) {
      humanGroups.push(`Le ${startStr} — ${g.hb} à ${g.he}`);
    } else {
      humanGroups.push(`Du ${startStr} au ${endStr} — ${g.hb} à ${g.he}`);
    }
  });

  return humanGroups;
}


function formatEvent(payload: EventPayload): string {
  const p = payload || {};

  const title = p.title_fr;
  const city = p.location_city || "Non précisé";
  const dep = p.location_department;
  const country = p.location_countrycode;
  const conditions = p.conditions_fr;
  const dates = (groupTimings(p.timings) || [])
    .map(line => `    ${line}`)
    .join("\n");

  return `
- ${title}
    📍 ${city} (${dep}, ${country})
    📅 Dates :
${dates}
    🎟 ${conditions ? conditions : "Conditions d'accès non précisées"}
  `.trim();
}

const ChatInputWithHistory: React.FC<{
  onSend: (text: string) => void;
  historyTexts: string[];
}> = ({ onSend, historyTexts }) => {
  const { history } = useUserHistory();
  const [input, setInput] = useState("");
  const { t } = useTranslation();

  // index in history we're currently showing
  // null = not navigating (user typing normally)
  const [histIndex, setHistIndex] = useState<number | null>(null);

  // keep a draft of what user was typing before pressing ↑
  const draftRef = useRef<string>("");

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const text = input.trim();
      onSend(text);
      setInput("");
      setHistIndex(null);
      draftRef.current = "";
      return;
    }

    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (historyTexts.length === 0) return;

      // first time pressing ↑: store draft + jump to last item
      if (histIndex === null) {
        draftRef.current = input;
        const last = historyTexts.length - 1;
        setHistIndex(last);
        setInput(historyTexts[last]);
        return;
      }

      // otherwise move up, clamp at 0
      const next = Math.max(0, histIndex - 1);
      setHistIndex(next);
      setInput(historyTexts[next]);
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (historyTexts.length === 0 || histIndex === null) return;

      const next = histIndex + 1;

      // if we pass the newest item, restore draft and exit nav mode
      if (next >= historyTexts.length) {
        setHistIndex(null);
        setInput(draftRef.current);
        return;
      }

      setHistIndex(next);
      setInput(historyTexts[next]);
      return;
    }
  };

  // if history changes (new request added), reset navigation safely
  useEffect(() => {
    setHistIndex(null);
  }, [history.length]);

  return (
    <Input
      placeholder={t("inputPlaceholder")}
      value={input}
      onChange={(e) => {
        setInput(e.target.value);
        if (histIndex !== null) setHistIndex(null); // exit history mode when typing
      }}
      onKeyDown={handleKeyDown}
      bg="gray.400"
      color="black"
      _placeholder={{ color: "gray.700" }}
    />
  );
};

const Chatbot: React.FC<ChatbotProps> = (
{
  endpoint,
}) => {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: t("welcomeMessage") },
  ]);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const { history, store } = useUserHistory();
  const historyTexts = history.map(h => h.text);
  const { city } = useCityFromGeolocation();

  const MessageRow: React.FC<{ message: Message }> = ({ message }) => {
    const isUser = message.role === "user";
    return (
        <HStack
          align="flex-start"
          justify={isUser ? "flex-end" : "flex-start"}
        >

        <Box
          bg={isUser ? "gray.400" : "white"}
          color={isUser ? "black" : "gray.800"}
          px={3}
          py={2}
          borderRadius="xl"
          boxShadow="sm"
          maxW="75%"
          whiteSpace="pre-wrap"
        >
          <Text fontSize="sm">{message.content}</Text>
        </Box>
      </HStack>
      )
    }

  const sendMessage = async (text: string) => {
  
    try {
      // Send to backend
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text
        }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const data = (await res.json()) as ChatResponse;

      // Add assistant response
      const items = data.answer.map(res => formatEvent(res.payload));
      const assistantMsg: Message = {
        role: "assistant",
        content: items.length > 0 ? items.join("\n\n") : t("searchResult"),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: t("connexionError")
        },
      ]);
    }
  };

  const search_events = async (text: string) => {
      const trimmed_text = text.trim();
      if (!trimmed_text)
        return;
      const userMsg: Message = { role: "user", content: trimmed_text };
      const history = [...messages, userMsg];
      setMessages(history);
      // Store user request in IndexedDB
      await store(text);
      // Send message to server
      sendMessage(text);
  }

  const search_on_geolocation = async () => {
    search_events(city + "?");
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);


  return (
    <Flex
      direction="column"
      flex="1"
      h="100vh"        // important: gives a fixed height to push bottom
      minH={0}         // important for proper scroll inside flex
      bg="gray.600"
    >
      <ChatbotHeader />

      {/* Scrollable content */}
      <Box flex="1" overflowY="auto" p={8}>
        {messages.map((m, i) => (
          <MessageRow key={i} message={m} />
        ))}
        <div ref={bottomRef} />
      </Box>

      {/* Stuck at bottom */}
      <Box
        mt="auto"
        p={4}
        borderTopWidth="0px"
      >
        <Flex direction="row">
          {city && (
          <IconButton aria-label="spark" bg="transparent" onClick={search_on_geolocation}>
            <span style={{ fontSize: "20px" }}>❇️</span>
          </IconButton>
          )}
          <ChatInputWithHistory onSend={search_events} historyTexts={historyTexts}/>
        </Flex>
      </Box>
    </Flex>
  );
};

export default Chatbot
