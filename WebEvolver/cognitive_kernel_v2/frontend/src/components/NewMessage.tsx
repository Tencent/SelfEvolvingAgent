import { Button, Typography } from "antd";
import { useContext } from "react";
import { v4 as uuidv4 } from "uuid";
import { Tooltip } from "antd";
import axios from "axios";
import { StatusContext } from "../contexts/StatusContext";
import { CKWSContext } from "../contexts/WebSocketConnectionContext";
import { FormOutlined } from "@ant-design/icons";

const { Title, Paragraph, Text, Link } = Typography;

function NewMessage({
  fetchHistory,
}: {
  fetchHistory: (model_name: string) => void;
}) {
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

  async function handleNewMessage() {
    const response = await axios.post("/api/clean_up_ck", {
      CKStatus: CKStatus,
      username: localStorage.getItem("username"),
    });
    setCKStatus((prevStatus) => ({
      ...prevStatus,
      messages: [],
      session_id: uuidv4(),
    }));
    fetchHistory(CKStatus.current_model);
  }
  return (
    <div style={{ display: "flex", justifyContent: "flex-end" }}>
      <Button
        type="text"
        onClick={handleNewMessage}
        style={{
          border: 0,
          boxShadow: "none",
          height: "40px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "0 16px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center" }}>
          <Text style={{ color: "#434343", fontSize: 24 }}>
            Cognitive Kernel
          </Text>
          <Tooltip title="New Session">
            <span
              style={{
                display: "inline-flex",
                marginLeft: 8,
                border: "1px dashed #d9d9d9",
                borderRadius: "50%",
                padding: "4px",
              }}
            >
              <FormOutlined />
            </span>
          </Tooltip>
        </div>
      </Button>
    </div>
  );
}

export default NewMessage;
