import React, { useState, useEffect } from "react";
import { Image } from "antd";
import MyAnnotator from "./MyAnnotator";
import { StatusContext } from "../contexts/StatusContext";

interface RenderSingleMessageContentProps {
  messageContent: string;
  currentMessageId: string;
  annotationActivated: boolean;
}

const fetchScreenshotWithBrowserID = async (
  browserId: string,
  pageId: string,
  currentRound: string
) => {
  try {
    const response = await fetch("/web/loadScreenshot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        browserId: browserId,
        pageId: pageId,
        currentRound: currentRound,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const blob = await response.blob();
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error("Failed to fetch screenshot:", error);
    throw error;
  }
};

const RenderSingleMessageContentWeb: React.FC<
  RenderSingleMessageContentProps
> = ({ messageContent, currentMessageId, annotationActivated }) => {
  const [imageSrcs, setImageSrcs] = useState<string[]>([]);

  const steps = messageContent.split(/\[WEB\]/).slice(1);

  useEffect(() => {
    const newImageSrcs = steps.map(async (step) => {
      const webPrefixPattern = /^\[([^\]]+)\] \[([^\]]+)\] \[([^\]]+)\] (.*)$/;
      const match = step.trim().match(webPrefixPattern);
      if (match && !match[4].trim().startsWith("stop")) {
        const [, browserId, pageId, currentRound] = match;
        try {
          const src = await fetchScreenshotWithBrowserID(
            browserId,
            pageId,
            currentRound
          );
          return src;
        } catch (error) {
          console.error("Failed to fetch screenshot:", error);
          return "";
        }
      } else {
        return "";
      }
    });

    Promise.all(newImageSrcs).then(setImageSrcs);
  }, [messageContent]);

  return (
    <>
      {steps.map((step, index) => {
        const webPrefixPattern =
          /^\[([^\]]+)\] \[([^\]]+)\] \[([^\]]+)\] (.*)$/;
        const match = step.trim().match(webPrefixPattern);
        if (match) {
          const [, browserID, pageID, currentRound, content] = match;
          const title = `Step ${index + 1}: ${content}`;
          const imageSrc = imageSrcs[index];

          let annotation_message_id =
            currentMessageId + "@@" + "web" + "@@" + index.toString();

          return (
            <>
              <div
                key={index}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <h3 style={{ margin: 0 }}>{title}</h3>{" "}
                {annotationActivated && index > 0 && (
                  <div>
                    <MyAnnotator messageId={annotation_message_id} />
                  </div>
                )}
              </div>
              {!content.trim().startsWith("stop") && imageSrc && (
                <Image
                  src={imageSrc}
                  alt={`Screenshot ${currentRound}`}
                  style={{ width: "100%" }}
                />
              )}
            </>
          );
        }
        return null;
      })}
    </>
  );
};

export default RenderSingleMessageContentWeb;
