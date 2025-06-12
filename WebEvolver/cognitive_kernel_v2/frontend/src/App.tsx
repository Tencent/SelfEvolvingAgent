import React, { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import { Layout, ConfigProvider } from "antd";
import axios from "axios";
import { CKStatusType, StatusContext } from "./contexts/StatusContext";
import {
  CKWSType,
  CKWSContextType,
  CKWSContext,
} from "./contexts/WebSocketConnectionContext";
import Messages from "./components/Messages";
import MySidebar from "./components/Sidebar";
import DeveloperAccess from "./components/MyAnnotation";
import ModelSelector from "./components/MyModelSelector";

const { Header, Content, Footer, Sider } = Layout;

const App: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [CKStatus, setCKStatus] = useState<CKStatusType>({
    current_model: "cognitiveKernel",
    activated_websites: [],
    activated_functions_personalized_model: false,
    activated_functions_self_improving: false,
    activated_functions_online_annotation: false,
    uploaded_files: [],
    messages: [],
    session_id: uuidv4(),
    history_id: uuidv4(),
    username: "",
    currentMode: "normal",
  });
  // localStorage.setItem("username", "NA");
  const [CKWSConnection, setCKWSConnection] = useState<CKWSType>({
    CKWSConnection: null,
  });
  const [historyMessages, setHistoryMessages] = useState([]);
  const [canSave, setCanSave] = useState(false);

  const fetchHistory = async (model_name: string) => {
    if (CKStatus.username === "") {
      setHistoryMessages([]);
    } else {
      let data_for_retrieval = {
        model_name: model_name,
        username: CKStatus.username,
      };
      const response = await axios.post(
        "/api/retrieve_history",
        data_for_retrieval
      );
      setHistoryMessages(response.data.data);
    }
  };

  useEffect(() => {
    fetchHistory(CKStatus.current_model);
  }, [CKStatus.username, CKStatus.current_model]);

  const SaveMessage = async () => {
    let data_for_saving = {
      username: localStorage.getItem("username"),
      session_id: CKStatus.session_id,
      model_name: CKStatus.current_model,
      messages: CKStatus.messages,
      updated_time: new Date().toISOString(),
    };
    try {
      const response = await axios.post(
        "/api/save_message_to_db",
        data_for_saving
      );
    } catch (error) {
      console.error("Error saving message", error);
    }
  };

  useEffect(() => {
    if (canSave) {
      SaveMessage();
      setCanSave(false);
    }
  }, [canSave]);

  useEffect(() => {
    fetchHistory(CKStatus.current_model);
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/setup_ws`;
    const my_websocket = new WebSocket(wsUrl);

    my_websocket.onopen = function (event) {
      console.log("WebSocket is open now.");
    };

    my_websocket.onmessage = function (event) {
      const data = event.data;
      // console.log("Received data from server:", data);

      if (data === "[save_message]") {
        console.log("Received [save_message] from server.");
        setCanSave(true);
        setCKStatus((prevStatus) => ({
          ...prevStatus,
          currentMode: "normal",
        }));
      } else if (data === "[task_cancelled]") {
        console.log("Received [task_cancelled] from server.");
        setCKStatus((prevStatus) => ({
          ...prevStatus,
          currentMode: "stopped",
        }));
      } else {
        try {
          const json = JSON.parse(data);
          setCKStatus((prevStatus) => {
            const newMessages = prevStatus.messages.slice();
            newMessages[newMessages.length - 1].message = json;
            return { ...prevStatus, messages: newMessages };
          });
        } catch (e) {
          console.error("Error parsing JSON:", e);
        }
      }
    };

    my_websocket.onerror = function (event) {
      console.error("WebSocket error:", event);
    };

    my_websocket.onclose = function (event) {
      console.log("WebSocket is closed now.");
    };
    setCKWSConnection({ CKWSConnection: my_websocket });
    return () => {
      my_websocket.close();
      console.log("WebSocket is closed now.");
    };
  }, []);

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider
        width={250}
        theme="light"
        collapsed={collapsed}
        onCollapse={(value) => {
          setCollapsed(value);
        }}
      >
        <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
          <CKWSContext.Provider value={{ CKWSConnection, setCKWSConnection }}>
            <MySidebar
              historyMessages={historyMessages}
              fetchHistory={fetchHistory}
            />
          </CKWSContext.Provider>
        </StatusContext.Provider>
      </Sider>
      <Layout style={{ position: "relative" }}>
        <div style={{ position: "absolute", top: 40, left: 40 }}>
          <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
            <ModelSelector fetchHistory={fetchHistory} />{" "}
          </StatusContext.Provider>
        </div>
        <Content
          style={{
            margin: "0 20%",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            height: "100%",
          }}
        >
          <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
            <CKWSContext.Provider value={{ CKWSConnection, setCKWSConnection }}>
              <div style={{ marginTop: "auto" }}>
                {" "}
                <Messages />
              </div>
            </CKWSContext.Provider>
          </StatusContext.Provider>
        </Content>
        <ConfigProvider
          theme={{
            token: {
              colorText: "rgba(0, 0, 0, 0.5)",
              fontSize: 10,
            },
          }}
        >
          <Footer
            style={{
              textAlign: "center",
              minHeight: "8vh",
            }}
          >
            Cognitive Kernel ©{new Date().getFullYear()} Created by Tencent AI
            Lab
          </Footer>
        </ConfigProvider>
      </Layout>

      <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
        <DeveloperAccess />
      </StatusContext.Provider>
    </Layout>
  );
};

export default App;
