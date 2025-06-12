import { List, Typography } from "antd";
import { useContext, useState } from "react";
import { StatusContext } from "../contexts/StatusContext";
import { CopyOutlined, DeleteOutlined } from "@ant-design/icons";
import moment from "moment";
import axios from "axios";

type HistoryMessage = {
  id: number;
  session_id: string;
  initial_message: string;
  updated_time: string;
};

const renderGroup = (
  messages: HistoryMessage[],
  title: string,
  fetchHistory: (model_name: string) => void
) => {
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;
  const [hoveredItemId, setHoveredItemId] = useState<number>(0);

  const handleClick = async (PrimarySessionId: number, SessionId: string) => {
    const response = await axios.post("/api/retrieve_message_session_by_id", {
      session_id: PrimarySessionId,
    });
    let selected_message = JSON.parse(response.data.data["messages"]);
    console.groupCollapsed(selected_message);
    setCKStatus((prevStatus) => ({
      ...prevStatus,
      messages: selected_message,
      session_id: SessionId,
    }));
  };

  const copyToClipboardFallback = (text: string) => {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      const successful = document.execCommand("copy");
      console.log(
        "Fallback: Copying text command was " +
          (successful ? "successful" : "unsuccessful")
      );
    } catch (err) {
      console.error("Fallback: Oops, unable to copy", err);
    }
    document.body.removeChild(textArea);
  };

  if (messages.length === 0) {
    return null;
  }

  const handleArchive = async (PrimarySessionId: number) => {
    const response = await axios.post("/api/archive_message_session_by_id", {
      session_id: PrimarySessionId,
    });
    if (response.data.status === "success") {
      fetchHistory(CKStatus.current_model);
    }
  };

  return (
    <List
      header={<div>{title}</div>}
      dataSource={messages}
      renderItem={(item) => (
        <List.Item
          onClick={() => handleClick(item.id, item.session_id)}
          onMouseEnter={() => setHoveredItemId(item.id)}
          onMouseLeave={() => setHoveredItemId(0)}
          actions={
            hoveredItemId === item.id
              ? [
                  <CopyOutlined
                    onClick={(e) => {
                      e.stopPropagation();
                      copyToClipboardFallback(item.initial_message);
                    }}
                    style={{
                      color: "darkgray",
                      fontSize: "15px",
                      marginTop: "1px",
                    }}
                  />,
                  <DeleteOutlined
                    onClick={(e) => {
                      e.stopPropagation();
                      handleArchive(item.id);
                    }}
                    style={{
                      color: "darkgray",
                      fontSize: "15px",
                      marginTop: "1px",
                    }}
                  />,
                ]
              : []
          }
          style={{
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Typography.Text ellipsis={{ tooltip: item.initial_message }}>
            {item.initial_message}
          </Typography.Text>
        </List.Item>
      )}
    />
  );
};

const HistoryList = ({
  historyMessages,
  fetchHistory,
}: {
  historyMessages: HistoryMessage[];
  fetchHistory: (model_name: string) => void;
}) => {
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;
  const groupMessagesByDate = (messages: HistoryMessage[]) => {
    const groups: {
      today: HistoryMessage[];
      yesterday: HistoryMessage[];
      lastWeek: HistoryMessage[];
      older: HistoryMessage[];
    } = {
      today: [],
      yesterday: [],
      lastWeek: [],
      older: [],
    };

    messages.forEach((message) => {
      const messageDate = moment(message.updated_time);
      if (messageDate.isSame(moment(), "day")) {
        groups.today.push(message);
      } else if (messageDate.isSame(moment().subtract(1, "days"), "day")) {
        groups.yesterday.push(message);
      } else if (messageDate.isAfter(moment().subtract(7, "days"))) {
        groups.lastWeek.push(message);
      } else {
        groups.older.push(message);
      }
    });

    return groups;
  };
  const groupedMessages = groupMessagesByDate(historyMessages);

  return (
    <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
      <div
        style={{
          height: "70vh",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          overflow: "auto",
        }}
      >
        {renderGroup(groupedMessages.today, "Today", fetchHistory)}
        {renderGroup(groupedMessages.yesterday, "Yesterday", fetchHistory)}
        {renderGroup(groupedMessages.lastWeek, "Last 7 Days", fetchHistory)}
        {renderGroup(groupedMessages.older, "Older", fetchHistory)}
      </div>
    </StatusContext.Provider>
  );
};

export default HistoryList;
