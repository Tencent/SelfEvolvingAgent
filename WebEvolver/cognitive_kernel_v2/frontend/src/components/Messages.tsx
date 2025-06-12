import React, { useEffect, useRef, useState, useContext } from "react";
import { v4 as uuidv4 } from "uuid";
import { List, Input, ConfigProvider } from "antd";
import { MyMessageType, StatusContext } from "../contexts/StatusContext";
import { CKStatusType } from "../contexts/StatusContext";
import { CKWSContext } from "../contexts/WebSocketConnectionContext";
import RenderItemComponent from "./MyMessageRender";
import FileUploader from "./FileUploader";
import axios from "axios";
import { SendOutlined, StopOutlined } from "@ant-design/icons";

type InferenceDataFormat = {
  role: string;
  content: string;
  name?: string;
};

function Messages() {
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;

  const CKWSConnectionContext = useContext(CKWSContext);

  if (!CKWSConnectionContext) {
    return <div>Context not available.</div>;
  }

  const { CKWSConnection, setCKWSConnection } = CKWSConnectionContext;

  const selectMessage = async (
    messages: MyMessageType[],
    messageId: string,
    messageSegmentGroup: string
  ) => {
    console.log("Selecting messages:", messages);
    console.log("Selecting message:", messageId);
    console.log("Selecting message segment group:", messageSegmentGroup);

    let messagesToKeep = [...messages];
    let targetMessageIndexReversed = [...messages]
      .reverse()
      .findIndex((m) => m.messageState.currentMessageId === messageId);
    let targetMessageIndex =
      targetMessageIndexReversed !== -1
        ? messages.length - 1 - targetMessageIndexReversed
        : -1;
    console.log("Target message index:", targetMessageIndex);
    if (messageSegmentGroup === "user_input") {
      messagesToKeep =
        targetMessageIndex !== -1
          ? messages.slice(0, targetMessageIndex - 1)
          : [...messages];
      let targetMessage = messages[targetMessageIndex - 1];
      let messageSegmentIndex = targetMessage.message.findIndex(
        (m) => m.group === messageSegmentGroup
      );
      targetMessage.message = [
        ...targetMessage.message.slice(0, messageSegmentIndex + 1),
      ];
      messagesToKeep = [...messagesToKeep, targetMessage];
    } else if (messageSegmentGroup === "assistant_slow_thinking") {
      messagesToKeep =
        targetMessageIndex !== -1
          ? messages.slice(0, targetMessageIndex)
          : [...messages];
      let targetMessage = messages[targetMessageIndex];
      let messageSegmentIndex = targetMessage.message.findIndex(
        (m) => m.group === messageSegmentGroup
      );
      targetMessage.message = [
        ...targetMessage.message.slice(0, messageSegmentIndex + 1),
      ];
      messagesToKeep = [...messagesToKeep, targetMessage];
    } else {
      console.error("Unknown message segment group:", messageSegmentGroup);
    }

    return messagesToKeep;
  };

  const handleReSend = async (
    messageId: string,
    messageSegmentGroup: string,
    messageContent?: string
  ) => {
    let messagesToKeep = await selectMessage(
      CKStatus.messages,
      messageId,
      messageSegmentGroup
    );
    console.log("Messages to keep:", messagesToKeep);

    let currentMessageId =
      messagesToKeep[messagesToKeep.length - 1].messageState.currentMessageId;
    if (messageContent) {
      var lastMessageToKeep = messagesToKeep[messagesToKeep.length - 1];
      var lastMessage =
        lastMessageToKeep.message[lastMessageToKeep.message.length - 1];
      lastMessage.content = messageContent;
    }
    let previousMessageId = "NA";
    if (messagesToKeep[messagesToKeep.length - 1].sender === "user") {
      if (messagesToKeep.length > 1) {
        previousMessageId =
          messagesToKeep[messagesToKeep.length - 2].messageState
            .currentMessageId || "NA";
      }
      const initial_message = {
        message: [{ group: "assistant_slow_thinking", content: "", pos: 0 }],
        messageState: {
          showEditBox: false,
          editText: "",
          currentMessageId: currentMessageId,
        },
        sender: CKStatus.current_model,
      };
      setCKStatus((prevStatus) => ({
        ...prevStatus,
        messages: [...messagesToKeep, initial_message],
        currentMode: "generation",
      }));
    } else {
      if (messagesToKeep.length > 2) {
        previousMessageId =
          messagesToKeep[messagesToKeep.length - 3].messageState
            .currentMessageId || "NA";
      }
      setCKStatus((prevStatus) => ({
        ...prevStatus,
        messages: messagesToKeep,
        currentMode: "generation",
      }));
    }

    let apiMessages = await fetchDataAndUpdate(previousMessageId);

    if (messagesToKeep[messagesToKeep.length - 1].sender === "user") {
      for (
        let i = 0;
        i < messagesToKeep[messagesToKeep.length - 1].message.length;
        i++
      ) {
        apiMessages.push({
          role: messagesToKeep[messagesToKeep.length - 1].message[
            i
          ].group.split("_")[0],
          content: messagesToKeep[messagesToKeep.length - 1].message[i].content,
        });
      }
    } else {
      for (
        let i = 0;
        i < messagesToKeep[messagesToKeep.length - 2].message.length;
        i++
      ) {
        apiMessages.push({
          role: messagesToKeep[messagesToKeep.length - 2].message[
            i
          ].group.split("_")[0],
          content: messagesToKeep[messagesToKeep.length - 2].message[i].content,
        });
      }
      for (
        let i = 0;
        i < messagesToKeep[messagesToKeep.length - 1].message.length;
        i++
      ) {
        apiMessages.push({
          role: messagesToKeep[messagesToKeep.length - 1].message[
            i
          ].group.split("_")[0],
          content: messagesToKeep[messagesToKeep.length - 1].message[i].content,
        });
      }
    }

    console.log("apiMessages:", apiMessages);
    console.log("currentMessageId:", currentMessageId);
    if (currentMessageId) {
      const apiRequestBody = {
        messages: apiMessages,
        CKStatus: CKStatus,
        username: localStorage.getItem("username"),
        currentMessageId: currentMessageId,
        action: "regeneration",
      };
      if (CKWSConnection.CKWSConnection) {
        CKWSConnection.CKWSConnection.send(JSON.stringify(apiRequestBody));
        console.log(
          "Sent data regeneration:",
          apiRequestBody,
          new Date().toISOString()
        );
      } else {
        console.error("WebSocket connection not available.");
      }
    }
  };

  const handleSend = async (message: string) => {
    let messagesToKeep = [...CKStatus.messages];
    if (CKStatus["currentMode"] === "stopped") {
      const lastUserMessageIndex = [...CKStatus.messages]
        .reverse()
        .findIndex((m) => m.sender === "user");

      const cutIndex =
        lastUserMessageIndex !== -1
          ? CKStatus.messages.length - 1 - lastUserMessageIndex
          : -1;

      messagesToKeep =
        cutIndex !== -1
          ? CKStatus.messages.slice(0, cutIndex)
          : [...CKStatus.messages];
      console.log("Messages to keep:", messagesToKeep);
    }
    let previousMessageId = "NA";
    if (messagesToKeep.length > 0) {
      previousMessageId =
        messagesToKeep[messagesToKeep.length - 1].messageState
          .currentMessageId || "NA";
    }

    let currentMessageId = uuidv4();
    const newMessage = {
      message: [{ group: "user_input", content: message, pos: 0 }],
      messageState: {
        showEditBox: false,
        editText: "",
        currentMessageId: currentMessageId,
      },
      sender: "user",
    };

    const newStatus = {
      ...CKStatus,
      messages: [...messagesToKeep, newMessage],
    };
    setCKStatus((prevStatus) => ({
      ...prevStatus,
      messages: [...messagesToKeep, newMessage],
    }));
    let apiMessages = await fetchDataAndUpdate(previousMessageId);
    apiMessages.push({ role: "user", content: message });
    await processMessagesToBackend(newStatus, apiMessages, currentMessageId);
  };

  async function fetchDataAndUpdate(previousMessageId: string) {
    let apiMessages: InferenceDataFormat[] = [];
    if (previousMessageId !== "NA") {
      try {
        const response = await axios.post(
          "/api/retrieve_rawdata_by_message_id",
          {
            currentMessageId: previousMessageId,
          }
        );
        let returned_result = JSON.parse(response.data.data[0]["raw_data"]);
        returned_result.shift();
        apiMessages = returned_result;
      } catch (error) {
        console.error("Error:", error);
      }
    } else {
      apiMessages = [];
    }

    return apiMessages;
  }

  async function processMessagesToBackend(
    status: CKStatusType,
    apiMessages: InferenceDataFormat[],
    currentMessageId: string
  ) {
    const apiRequestBody = {
      messages: apiMessages,
      CKStatus: status,
      username: localStorage.getItem("username"),
      currentMessageId: currentMessageId,
      action: "generation",
    };

    const initialCkMessage = {
      message: [{ group: "assistant_slow_thinking", content: "", pos: 0 }],
      messageState: {
        showEditBox: false,
        editText: "",
        currentMessageId: currentMessageId,
      },
      sender: CKStatus.current_model,
    };
    setCKStatus((prevStatus) => ({
      ...prevStatus,
      messages: [...prevStatus.messages, initialCkMessage],
      currentMode: "generation",
    }));
    if (CKWSConnection.CKWSConnection) {
      CKWSConnection.CKWSConnection.send(JSON.stringify(apiRequestBody));
      console.log("Sent data:", apiRequestBody, new Date().toISOString());
    } else {
      console.error("WebSocket connection not available.");
    }
  }

  async function stopGeneration() {
    const apiRequestBody = {
      action: "stop",
    };
    if (CKWSConnection.CKWSConnection) {
      CKWSConnection.CKWSConnection.send(JSON.stringify(apiRequestBody));
      console.log("Sent data:", apiRequestBody, new Date().toISOString());
    } else {
      console.error("WebSocket connection not available.");
    }
  }
  const [inputValue, setInputValue] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && !isComposing) {
      handlePressEnter();
    }
  };
  const handleCompositionStart = () => {
    setIsComposing(true);
  };
  const handleCompositionEnd = () => {
    setIsComposing(false);
  };
  const handlePressEnter = () => {
    handleSend(inputValue);
    setInputValue("");
  };
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [CKStatus.messages]);

  return (
    <>
      <ConfigProvider
        theme={{
          token: {
            colorTextDescription: "#5C5C5C",
          },
        }}
      >
        <List
          itemLayout="horizontal"
          style={{
            margin: "60px 0 0 0",
            height: "80vh",
            overflow: "auto",
          }}
          dataSource={CKStatus.messages}
          renderItem={(item, index) => (
            <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
              <RenderItemComponent
                item={item}
                index={index}
                key={index}
                handleReSend={handleReSend}
              />
            </StatusContext.Provider>
          )}
        >
          <div ref={listRef} />
        </List>
      </ConfigProvider>

      <Input
        style={{ marginTop: "20px" }}
        placeholder="Message Cognitive Kernel"
        value={inputValue}
        prefix={<FileUploader />}
        suffix={
          CKStatus["currentMode"] === "generation" ? (
            <StopOutlined onClick={stopGeneration} />
          ) : (
            <SendOutlined onClick={handlePressEnter} />
          )
        }
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onCompositionStart={handleCompositionStart}
        onCompositionEnd={handleCompositionEnd}
      />
    </>
  );
}

export default Messages;
