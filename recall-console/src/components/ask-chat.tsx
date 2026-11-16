"use client";
import { useState } from "react";
import { IllosAiChatEmptyState } from "@/components/ui/illos-ai-chat-empty-state";
import SiriOrb from "@/components/ui/siri-orb";

type Msg = { role: "user" | "assistant"; text: string; error?: boolean };

const SUGGESTIONS = [
  "Does Global's fix work on Fire in Urdu?",
  "Which relatives does the family fix not help?",
  "How does the family fix compare with rollback on Earth?",
];

export function AskChat() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setMsgs((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    setDraft("");
    try {
      const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
      const j = await r.json();
      setMsgs((m) => [...m, r.ok ? { role: "assistant", text: j.answer } : { role: "assistant", text: j.error, error: true }]);
    } catch {
      setMsgs((m) => [...m, { role: "assistant", text: "Could not reach the server. Check your connection and ask again.", error: true }]);
    } finally {
      setBusy(false);
    }
  }

  if (!msgs.length)
    return (
      <IllosAiChatEmptyState
        title="Ask the evidence"
        description="Answers come only from the measured results in this console, with the fact they came from in brackets."
        suggestions={SUGGESTIONS}
        placeholder="Ask about a sibling, a language or a relative"
        onSubmit={ask}
      />
    );

  return (
    <div className="flex flex-col gap-4">
      {msgs.map((m, i) => (
        <div key={i} className={m.role === "user" ? "self-end max-w-[85%] rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground" : `max-w-[85%] rounded-lg border bg-card px-4 py-3 text-sm leading-relaxed ${m.error ? "border-affected/60" : ""}`}>
          {m.text}
        </div>
      ))}
      {busy && (
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <SiriOrb size="28px" animationDuration={6} /> Reading the results
        </div>
      )}
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          ask(draft);
        }}
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask a follow-up"
          aria-label="Question"
          className="flex-1 rounded-md border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <button type="submit" disabled={busy} className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">
          Ask
        </button>
      </form>
    </div>
  );
}
