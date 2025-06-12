import { Divider, Flex } from "antd";
import NewMessage from "./NewMessage";
import HistoryList from "./HistoryList";
import MyLogin from "./MyLogin";
import { useContext } from "react";
import { StatusContext } from "../contexts/StatusContext";
import { CKWSContext } from "../contexts/WebSocketConnectionContext";

type HistoryMessage = {
  id: number;
  session_id: string;
  initial_message: string;
  updated_time: string;
};

interface MySidebarProps {
  historyMessages: HistoryMessage[];
  fetchHistory: (model_name: string) => Promise<void>;
}

function MySidebar({ historyMessages, fetchHistory }: MySidebarProps) {
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;

  const CKWSConnectionContext = useContext(CKWSContext);

  if (!CKWSConnectionContext) {
    return <div>CKWS Connection Context not available.</div>;
  }

  const { CKWSConnection, setCKWSConnection } = CKWSConnectionContext;

  return (
    <>
      <div className="demo-logo-vertical">
        <div
          style={{
            margin: "24px 10px 0",
            display: "flex",
          }}
        >
          <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
            <CKWSContext.Provider value={{ CKWSConnection, setCKWSConnection }}>
              <NewMessage fetchHistory={fetchHistory} />
            </CKWSContext.Provider>
          </StatusContext.Provider>
        </div>

        <div
          style={{
            margin: "12px 12px 0",
            display: "flex",
          }}
        >
          <Divider
            dashed
            style={{
              borderColor: "#707070",
              borderStyle: "dashed",
              borderWidth: "0.5px",
            }}
          />
        </div>

        <div
          style={{
            margin: "0 24px",
            display: "flex",
            width: "80%",
            height: "70%",
          }}
        >
          <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
            <HistoryList
              historyMessages={historyMessages}
              fetchHistory={fetchHistory}
            />
          </StatusContext.Provider>
        </div>

        <Flex
          vertical
          gap="small"
          style={{
            margin: "0 24px",
            width: "80%",
            position: "absolute",
            bottom: 20,
            left: 0,
          }}
        >
          <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
            <MyLogin />
          </StatusContext.Provider>
        </Flex>
      </div>
    </>
  );
}

export default MySidebar;
