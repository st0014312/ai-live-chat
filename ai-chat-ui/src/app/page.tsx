"use client";

import { useStream } from "@langchain/langgraph-sdk/react";

export default function Home() {
  const thread = useStream({
    apiUrl: "http://localhost:8000",
    assistantId: "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
    messagesKey: "messages",
  });
  return (
    <div>
      <div>
        {thread.messages.map((message) => (
          <div key={message.id}>{message.content as string}</div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();

          const form = e.target as HTMLFormElement;
          const message = new FormData(form).get("message") as string;

          form.reset();
          thread.submit({ messages: [{ type: "human", content: message }] });
        }}
      >
        <input type="text" name="message" />

        {thread.isLoading ? (
          <button type="button" onClick={() => thread.stop()}>
            Stop
          </button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>
    </div>
  );
}
