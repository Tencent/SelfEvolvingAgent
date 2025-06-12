import React, { useState, useEffect } from "react";
import MyCodeBlock from "./MyCodeBlock";

interface MessagePart {
  type: "text" | "code";
  language: string;
  content: string;
}

function parseMessage(message: string): MessagePart[] {
  const regex = /```/g;
  let parts: MessagePart[] = [];
  let lastIndex = 0;
  let isCodeBlock = false;
  let codeLanguage = "";
  let codeLanguageLength = 0;

  message.replace(regex, (match, index) => {
    if (isCodeBlock) {
      parts.push({
        type: "code",
        language: codeLanguage,
        content: message.slice(lastIndex + codeLanguageLength + 1, index),
      });
      isCodeBlock = false;
      codeLanguage = "";
    } else {
      parts.push({
        type: "text",
        language: "",
        content: message.slice(lastIndex, index),
      });
      isCodeBlock = true;

      const languageIdentifier = message
        .slice(index + match.length)
        .match(/^[^\n]+/);
      codeLanguage = languageIdentifier
        ? languageIdentifier[0].trim()
        : "unknown";
      codeLanguageLength = languageIdentifier
        ? languageIdentifier[0].length
        : 0;
    }
    lastIndex = index + match.length;
    return match;
  });

  if (lastIndex < message.length) {
    if (isCodeBlock) {
      parts.push({
        type: "code",
        language: codeLanguage,
        content: message.slice(lastIndex),
      });
    } else {
      parts.push({
        type: "text",
        language: "",
        content: message.slice(lastIndex),
      });
    }
  }

  return parts;
}

interface RenderSingleMessageContentProps {
  messageContent: string;
}
const RenderSingleMessageContent: React.FC<RenderSingleMessageContentProps> = ({
  messageContent,
}) => {
  const messageParts = parseMessage(messageContent);
  return (
    <>
      {messageParts.map((part, partIndex) =>
        part.type === "code" ? (
          <MyCodeBlock
            key={partIndex}
            language={part.language}
            code={part.content}
          />
        ) : (
          <span key={partIndex}>{part.content}</span>
        )
      )}
    </>
  );
};

export default RenderSingleMessageContent;
