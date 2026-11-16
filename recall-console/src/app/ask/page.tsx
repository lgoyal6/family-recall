import { AskChat } from "@/components/ask-chat";

export default function AskPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-10 md:px-10">
      <AskChat />
      <p className="mt-8 text-xs text-muted-foreground">
        Answered by GPT-5.6 Luna through GMI Cloud, given only this console&apos;s result files. It can be wrong; every claim names the
        fact it came from so you can check it on the other pages.
      </p>
    </div>
  );
}
