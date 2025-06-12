import React, { useState, useContext } from "react";
import {
  Button,
  Drawer,
  message,
  Input,
  Dropdown,
  Space,
  Checkbox,
} from "antd";
import { MenuProps } from "antd";
import { DownOutlined } from "@ant-design/icons";
import ReactJson, { InteractionProps } from "react-json-view";
import axios from "axios";
import { CheckboxChangeEvent } from "antd/es/checkbox";
import { StatusContext } from "../contexts/StatusContext";

interface AnnotationProps {
  messageId: string;
}

function toLocalISOString(date: Date): string {
  const offset = date.getTimezoneOffset() * 60000;
  const localDate = new Date(date.getTime() - offset);
  const iso = localDate.toISOString();
  return iso;
}

interface tagProps {
  key: string;
  label: string;
}

const MyAnnotator: React.FC<AnnotationProps> = ({ messageId }) => {
  const [showDrawer, setShowDrawer] = useState(false);
  const [editText, setEditText] = useState({});
  const [sessionId, setSessionId] = useState("");
  const [oldKnowledge, setOldKnowledge] = useState("");
  const [suggestion, setSuggestion] = useState("");
  const [selectedTag, setSelectedTag] = useState("");
  const [forEvaluation, setForEvaluation] = useState(false);
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }
  const { CKStatus, setCKStatus } = context;

  const fetchMessageData = () => {
    if (messageId) {
      axios
        .post("/api/retrieve_rawdata_by_message_id", {
          currentMessageId: messageId,
        })
        .then((response) => {
          setShowDrawer(true);
          setOldKnowledge(response.data.data[0]["raw_data"]);
          setEditText(JSON.parse(response.data.data[0]["raw_data"]));
          setSessionId(response.data.data[0]["session_id"]);
        })
        .catch((error) => {
          console.error("Error fetching message data:", error);
        });
    }
  };

  const handleJsonChange = (edit: InteractionProps) => {
    setEditText(edit.updated_src);
  };

  const handleSubmit = () => {
    axios
      .post("/api/submit_annotation", {
        session_id: sessionId,
        currentMessageId: messageId,
        username: CKStatus.username,
        character_name: CKStatus.current_model,
        tag: selectedTag,
        for_evaluation: forEvaluation,
        oldKnowledge: oldKnowledge,
        messages_in_train_format: JSON.stringify(editText),
        Suggestion: suggestion,
        updated_time: toLocalISOString(new Date()),
      })
      .then(() => {
        message.success("Annotation submitted successfully");
        setShowDrawer(false);
      })
      .catch((error) => {
        message.error("Annotation submission failed");
        console.error("Submit error:", error);
      });
  };

  const tag_candidates: tagProps[] = [
    {
      key: "1",
      label: "Normal",
    },
    {
      key: "2",
      label: "DocAgent",
    },
    {
      key: "3",
      label: "WebAgent",
    },
    {
      key: "4",
      label: "PersonalHistory",
    },
    {
      key: "5",
      label: "Mix",
    },
  ];

  const onTagSelectionClick: MenuProps["onClick"] = ({ key }) => {
    const clickedItem = tag_candidates.find((item) => item.key === key);

    if (clickedItem !== undefined) {
      setSelectedTag(clickedItem.label);
    } else {
      setSelectedTag("");
    }
  };

  const tagMenuConfig = {
    items: tag_candidates,
    onClick: onTagSelectionClick,
  };

  const selectForEvaluation = (e: CheckboxChangeEvent) => {
    setForEvaluation(e.target.checked);
  };

  return (
    <>
      <Button onClick={fetchMessageData}>Annotate</Button>
      {showDrawer && (
        <Drawer
          title="Online Data Annotator"
          onClose={() => setShowDrawer(false)}
          open={showDrawer}
          width={720}
        >
          <ReactJson
            src={editText}
            onEdit={handleJsonChange}
            onAdd={handleJsonChange}
            onDelete={handleJsonChange}
            theme="rjv-default"
            style={{
              height: "70vh",
              border: "2px solid #ccc",
              padding: "10px",
              borderRadius: "5px",
              marginTop: "10px",
              marginBottom: "10px",
              overflow: "auto",
            }}
          />
          <Input
            placeholder="Please type in the suggestion for this annotation."
            value={suggestion}
            onChange={(e) => setSuggestion(e.target.value)}
            style={{ marginTop: "20px" }}
          />
          <div
            style={{
              marginTop: "20px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <Dropdown menu={tagMenuConfig} trigger={["click"]}>
              <a onClick={(e) => e.preventDefault()} style={{ color: "black" }}>
                <Space>
                  {selectedTag !== "" ? selectedTag : "Please select a tag"}
                  <DownOutlined />
                </Space>
              </a>
            </Dropdown>
            <Checkbox onChange={selectForEvaluation}>Used for testing</Checkbox>
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              marginTop: "20px",
            }}
          >
            <Button type="primary" onClick={handleSubmit}>
              Submit
            </Button>
          </div>
        </Drawer>
      )}
    </>
  );
};

export default MyAnnotator;
