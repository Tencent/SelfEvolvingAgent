import React, { useContext, useState } from "react";
import {
  MyMessageSegmentationType,
  MyMessageType,
} from "../contexts/StatusContext";
import { Avatar, List, Collapse, Button, Input, Tooltip } from "antd";

import { StatusContext } from "../contexts/StatusContext";
import { RedoOutlined, EditOutlined } from "@ant-design/icons";
const { TextArea } = Input;

import userImage from "../assets/user.png";
import RenderSingleMessageContent from "./MySingleMessageRender";
import RenderSingleMessageContentWeb from "./MySingleMessageRenderWeb";
import MyAnnotator from "./MyAnnotator";

interface RenderSegmentationProps {
  messageSegmentation: MyMessageSegmentationType;
  currentMessageId: string;
  annotationActivated: boolean;
  handleReSend: (
    messageId: string,
    messageSegmentGroup: string,
    messageContent?: string
  ) => void;
  currentMode: string;
}

const RenderSegmentation: React.FC<RenderSegmentationProps> = ({
  messageSegmentation,
  currentMessageId,
  annotationActivated,
  handleReSend,
  currentMode,
}) => {
  let RenderSingleMessageSelected = RenderSingleMessageContent;
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState("");

  const activateEditing = () => {
    setEditText(messageSegmentation.content);
    setIsEditing(true);
  };

  const editAndResend = (segmentGroup: string) => {
    const editContent = editText;
    handleReSend(currentMessageId, segmentGroup, editContent);
    setIsEditing(false);
    setEditText("");
  };

  const cancelEditting = () => {
    setIsEditing(false);
    setEditText("");
  };

  const key = `segment-${messageSegmentation.group}-${messageSegmentation.pos}`;

  if (messageSegmentation.group === "assistant_slow_thinking") {
    const messageItems = [
      {
        key: `panel-${key}`,
        label: "Slow Thinking",
        children: (
          <RenderSingleMessageSelected
            messageContent={messageSegmentation.content}
          />
        ),
        // children: (
        //   <div key={key}>
        //     {isEditing ? (
        //       <TextArea
        //         value={editText}
        //         onChange={(e) => setEditText(e.target.value)}
        //         placeholder=""
        //         autoSize={{ minRows: 1, maxRows: 50 }}
        //         style={{ width: "100%" }}
        //       />
        //     ) : (
        //       <RenderSingleMessageSelected
        //         messageContent={messageSegmentation.content}
        //       />
        //     )}
        //     {(currentMode === "normal" || currentMode === "stopped") &&
        //       isEditing === true && (
        //         <div
        //           style={{
        //             display: "flex",
        //             justifyContent: "center",
        //             alignItems: "center",
        //             marginTop: "10px",
        //           }}
        //         >
        //           <Button
        //             key="downloadCore"
        //             onClick={() => editAndResend(messageSegmentation.group)}
        //             style={{ marginRight: "10px" }}
        //           >
        //             Regenerate
        //           </Button>
        //           <Button
        //             key="downloadRaw"
        //             type="dashed"
        //             onClick={() => cancelEditting()}
        //           >
        //             Cancel
        //           </Button>
        //         </div>
        //       )}
        //     {(currentMode === "normal" || currentMode === "stopped") &&
        //       isEditing === false && (
        //         <>
        //           <br />
        //           <Tooltip title="Regenerate">
        //             <RedoOutlined
        //               onClick={() =>
        //                 currentMessageId
        //                   ? handleReSend(
        //                       currentMessageId,
        //                       messageSegmentation.group
        //                     )
        //                   : null
        //               }
        //               style={{
        //                 color: "darkgray",
        //                 fontSize: "8px",
        //                 marginTop: "1px",
        //               }}
        //             />
        //           </Tooltip>
        //           <Tooltip title="Modification">
        //             <EditOutlined
        //               onClick={activateEditing}
        //               style={{
        //                 color: "darkgray",
        //                 fontSize: "8px",
        //                 marginLeft: "10px",
        //               }}
        //             />
        //           </Tooltip>
        //         </>
        //       )}
        //   </div>
        // ),
      },
    ];
    return (
      <Collapse
        bordered={false}
        key={key}
        size="small"
        ghost
        defaultActiveKey={["1"]}
        items={messageItems}
      />
    );
  } else if (messageSegmentation.group === "assistant_execution_result") {
    const messageItems = [
      {
        key: `panel-${key}`,
        label: "Execution Result",
        children: (
          <RenderSingleMessageContent
            messageContent={messageSegmentation.content}
          />
        ),
      },
    ];
    return (
      <Collapse
        bordered={false}
        key={key}
        size="small"
        ghost
        defaultActiveKey={["1"]}
        items={messageItems}
      />
    );
  } else if (messageSegmentation.group === "assistant_web_result") {
    const messageItems = [
      {
        key: `panel-${key}`,
        label: "Web Exploration",
        children: currentMessageId && (
          <RenderSingleMessageContentWeb
            messageContent={messageSegmentation.content}
            currentMessageId={currentMessageId}
            annotationActivated={annotationActivated}
          />
        ),
      },
    ];
    return (
      <Collapse
        bordered={false}
        key={key}
        size="small"
        ghost
        defaultActiveKey={["1"]}
        items={messageItems}
      />
    );
  } else if (messageSegmentation.group === "assistant_final_output") {
    return (
      <div key={key}>
        <RenderSingleMessageSelected
          messageContent={messageSegmentation.content}
        />
      </div>
    );
  } else if (messageSegmentation.group === "user_input") {
    return (
      <div key={key}>
        {isEditing ? (
          <TextArea
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            placeholder=""
            autoSize={{ minRows: 1, maxRows: 50 }}
            style={{ width: "100%" }}
          />
        ) : (
          <RenderSingleMessageSelected
            messageContent={messageSegmentation.content}
          />
        )}
        {(currentMode === "normal" || currentMode === "stopped") &&
          isEditing === true && (
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                marginTop: "10px",
              }}
            >
              <Button
                key="downloadCore"
                onClick={() => editAndResend(messageSegmentation.group)}
                style={{ marginRight: "10px" }}
              >
                Regenerate
              </Button>
              <Button
                key="downloadRaw"
                type="dashed"
                onClick={() => cancelEditting()}
              >
                Cancel
              </Button>
            </div>
          )}
        {(currentMode === "normal" || currentMode === "stopped") &&
          isEditing === false && (
            <>
              <br />
              <Tooltip title="Regenerate">
                <RedoOutlined
                  onClick={() =>
                    currentMessageId
                      ? handleReSend(
                          currentMessageId,
                          messageSegmentation.group
                        )
                      : null
                  }
                  style={{
                    color: "darkgray",
                    fontSize: "8px",
                    marginTop: "1px",
                  }}
                />
              </Tooltip>
              <Tooltip title="Cancel">
                <EditOutlined
                  onClick={activateEditing}
                  style={{
                    color: "darkgray",
                    fontSize: "8px",
                    marginLeft: "10px",
                  }}
                />
              </Tooltip>
            </>
          )}
      </div>
    );
  } else {
    console.error("Unknown message group: ", messageSegmentation.group);
    return (
      <div key={key}>
        <RenderSingleMessageSelected
          messageContent={messageSegmentation.content}
        />
      </div>
    );
  }
};

const renderSingleMessageByGroup = (
  message: MyMessageType,
  annotation_activated: boolean,
  handleReSend: (
    messageId: string,
    messageSegmentGroup: string,
    messageContent?: string
  ) => void,
  currentMode: string
) => {
  let messageContent = message.message;
  const sortedMessageGroups = messageContent.sort((a, b) => a.pos - b.pos);
  const currentMessageId = message.messageState.currentMessageId;
  if (!currentMessageId) {
    return <div>Message ID is null</div>;
  }

  return sortedMessageGroups.map((messageSegmentation) => (
    <RenderSegmentation
      key={`segmentation-${messageSegmentation.pos}`}
      messageSegmentation={messageSegmentation}
      currentMessageId={currentMessageId}
      annotationActivated={annotation_activated}
      handleReSend={handleReSend}
      currentMode={currentMode}
    />
  ));
};

const RenderItemComponent = ({
  item,
  index,
  handleReSend,
}: {
  item: MyMessageType;
  index: number;
  handleReSend: (
    messageId: string,
    messageSegmentGroup: string,
    messageContent?: string
  ) => void;
}) => {
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;

  const renderItem = (message: MyMessageType) => {
    return (
      <>
        <List.Item
          style={{
            paddingRight: "10px",
            position: "relative",
            minHeight: "60px",
          }}
        >
          <List.Item.Meta
            avatar={
              <Avatar
                src={
                  message.sender === "user"
                    ? userImage
                    : `/api/avatar/${CKStatus.current_model}.png`
                }
              />
            }
            title={message.sender === "user" ? "You" : message.sender}
            description={renderSingleMessageByGroup(
              message,
              CKStatus.activated_functions_online_annotation,
              handleReSend,
              CKStatus.currentMode
            )}
          />
          {message.sender !== "user" &&
            CKStatus.activated_functions_online_annotation &&
            message.messageState.currentMessageId && (
              <div
                style={{ position: "absolute", top: 0, right: 0, zIndex: 2 }}
              >
                <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
                  <MyAnnotator
                    messageId={message.messageState.currentMessageId}
                  />
                </StatusContext.Provider>
              </div>
            )}
        </List.Item>
      </>
    );
  };

  return renderItem(item);
};

export default RenderItemComponent;
