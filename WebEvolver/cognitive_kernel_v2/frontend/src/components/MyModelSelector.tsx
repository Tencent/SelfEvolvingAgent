import React, { useState, useContext, useEffect } from "react";
import {
  Tooltip,
  Dropdown,
  Modal,
  Space,
  Form,
  Upload,
  Input,
  Button,
  message,
} from "antd";
import { MenuProps } from "antd";
import axios from "axios";
import {
  DownOutlined,
  PlusOutlined,
  UploadOutlined,
  EditOutlined,
  DeleteOutlined,
} from "@ant-design/icons";
import { StatusContext } from "../contexts/StatusContext";
import { v4 as uuidv4 } from "uuid";
import type { GetProp, UploadFile, UploadProps } from "antd";
import ImgCrop from "antd-img-crop";

type AvatarFileType = Parameters<GetProp<UploadProps, "beforeUpload">>[0];

interface MyModelSelectorProps {
  fetchHistory: (model_name: string) => Promise<void>;
}

interface receivedModelItem {
  name: string;
  key: string;
  shownTitle: string;
  title: string;
  characterType: string;
}
interface ModelItem {
  content: JSX.Element;
  name: string;
  key: string;
  title: string;
  characterType: string;
}
function ModelSelector({ fetchHistory }: MyModelSelectorProps) {
  const context = useContext(StatusContext);
  const [isCreatingNewAgent, setIsCreatingNewAgent] = useState(false);
  if (!context) {
    return <div>Context not available.</div>;
  }
  const { CKStatus, setCKStatus } = context;
  const [hoveredItemId, setHoveredItemId] = useState<string>("-1");
  const [visibleItems, setVisibleItems] = useState<ModelItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<React.ReactNode>(null);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [avatarURL, setAvatarURL] = useState("");
  const [backgroundURL, setBackgroundURL] = useState("");
  const fetchCharacters = async () => {
    try {
      const response = await axios.post("/api/get_all_characters", null, {
        params: { username: CKStatus.username },
      });
      const data: receivedModelItem[] = response.data;

      const itemsWithAvatar = await Promise.all(
        data.map(async (character) => {
          return {
            content: (
              <Space align="center" style={{ alignItems: "center" }}>
                <img
                  src={`/api/avatar/${character.name}.png`}
                  alt=""
                  style={{ width: 30, height: 30 }}
                />
                <span style={{ fontSize: "18px", lineHeight: "30px" }}>
                  {character.shownTitle}
                </span>
              </Space>
            ),
            name: character.name,
            key: character.characterType + "_" + character.key,
            title: character.title,
            characterType: character.characterType,
          };
        })
      );

      itemsWithAvatar.sort((a, b) => {
        if (a.characterType === b.characterType) {
          return a.key.localeCompare(b.key);
        }
        return a.characterType === "global" ? -1 : 1;
      });

      const defaultItem = itemsWithAvatar.find(
        (item) => item.name === CKStatus.current_model
      );
      if (defaultItem) {
        setSelectedItem(defaultItem.content);
      } else {
        const ck = itemsWithAvatar.find(
          (item) => item.name === "cognitiveKernel"
        );
        if (ck) {
          setCKStatus((prevStatus) => ({
            ...prevStatus,
            current_model: ck.name,
            messages: [],
            session_id: uuidv4(),
          }));
          fetchHistory(ck.name);
          setSelectedItem(ck.content);
        }
      }
      setVisibleItems(itemsWithAvatar);
    } catch (error) {
      console.error("Failed to fetch characters", error);
    }
  };
  useEffect(() => {
    fetchCharacters();
  }, [CKStatus.username]);
  const onClick: MenuProps["onClick"] = ({ key }) => {
    if (key === "createAgent") {
      setIsCreatingNewAgent(true);
    } else {
      const clickedItem = visibleItems.find((item) => item.key === key);
      if (clickedItem) {
        setSelectedItem(clickedItem.content);
        setCKStatus((prevStatus) => ({
          ...prevStatus,
          current_model: clickedItem.name,
          messages: [],
          session_id: uuidv4(),
        }));
        fetchHistory(clickedItem.name);
      }
    }
  };

  const handleEdit = (key: string) => {
    message.info("We will support this feature soon!");
    console.log("Edit", key);
  };
  const handleDelete = async (key: string) => {
    const formData = new FormData();
    formData.append("username", CKStatus.username);
    const clickedItem = visibleItems.find((item) => item.key === key);
    if (!clickedItem) {
      message.error("Agent Delete Failed");
    } else {
      formData.append("agent_name", clickedItem.name.split("_")[1]);
      try {
        const response = await axios.post("/api/delete_agent/", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        message.success("Agent Deleted Successfully");
        fetchCharacters();
      } catch (error) {
        console.error("Failed to delete Agent:", error);
        message.error("Agent Delete Failed");
      }
      fetchCharacters();
    }
  };

  const menuConfig = {
    items: [
      ...visibleItems.map((item) => {
        return {
          key: item.key,
          label: (
            <div
              onMouseEnter={() => setHoveredItemId(item.key)}
              onMouseLeave={() => setHoveredItemId("-1")}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <Tooltip title={item.title}>
                <span
                  style={{ display: "flex", alignItems: "center", flex: 1 }}
                >
                  {item.content}
                </span>
              </Tooltip>
              {item.characterType === "customized" &&
                hoveredItemId === item.key && (
                  <span style={{ display: "flex", justifyContent: "flex-end" }}>
                    <EditOutlined
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEdit(item.key);
                      }}
                      style={{
                        color: "darkgray",
                        fontSize: "18px",
                        marginTop: "1px",
                        marginRight: "8px",
                      }}
                    />
                    <DeleteOutlined
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(item.key);
                      }}
                      style={{
                        color: "darkgray",
                        fontSize: "18px",
                        marginTop: "1px",
                      }}
                    />
                  </span>
                )}
            </div>
          ),
        };
      }),
      {
        key: "createAgent",
        label: (
          <span
            style={{ display: "flex", alignItems: "center", fontSize: "18px" }}
          >
            <PlusOutlined
              style={{
                border: "1px dashed",
                borderRadius: "50%",
                marginRight: "8px",
                fontSize: "28px",
              }}
            />
            Create Agent
          </span>
        ),
      },
    ],
    onClick: onClick,
  };

  const closeNewAgentCreation = () => {
    setIsCreatingNewAgent(false);
  };

  const [form] = Form.useForm();

  const handleAgentCreation = async (values: any) => {
    const formData = new FormData();
    formData.append("username", CKStatus.username);
    formData.append("avatarURL", avatarURL);
    formData.append("backgroundURL", backgroundURL);
    formData.append("agent_name", values.name);
    formData.append("agent_id", values.id);
    formData.append("description", values.description);
    try {
      const response = await axios.post("/api/create_agent/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      message.success("Agent Created Successfully");
      fetchCharacters();
    } catch (error) {
      console.error("Failed to create Agent:", error);
      message.error("Agent Creation Failed");
    }
    form.resetFields();
    closeNewAgentCreation();
    fetchCharacters();
  };

  const onAvatarChange: UploadProps["onChange"] = ({
    fileList: newFileList,
  }) => {
    if (
      newFileList.length > 0 &&
      newFileList[0].response &&
      newFileList[0].response["url"]
    ) {
      setAvatarURL(newFileList[0].response["url"]);
    }
    setFileList(newFileList);
  };

  const onAvatarPreview = async (file: UploadFile) => {
    let src = file.url as string;
    if (!src) {
      src = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.readAsDataURL(file.originFileObj as AvatarFileType);
        reader.onload = () => resolve(reader.result as string);
      });
    }
    const image = new Image();
    image.src = src;
    const imgWindow = window.open(src);
    imgWindow?.document.write(image.outerHTML);
  };

  const characterBackgroundProps: UploadProps = {
    name: "file",
    action: "/api/upload_character_info",
    headers: {
      authorization: "authorization-text",
    },
    accept: ".txt",
    beforeUpload(file) {
      const isTxt = file.type === "text/plain";
      if (!isTxt) {
        message.error("You can only upload TXT file!");
      }
      return isTxt || Upload.LIST_IGNORE;
    },
    onChange(info) {
      if (info.file.response && info.file.response["url"]) {
        setBackgroundURL(info.file.response["url"]);
      }
      if (info.file.status !== "uploading") {
        console.log(info.file, info.fileList);
      }
      if (info.file.status === "done") {
        message.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === "error") {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    progress: {
      strokeColor: {
        "0%": "#108ee9",
        "100%": "#87d068",
      },
      strokeWidth: 3,
      format: (percent) => percent && `${parseFloat(percent.toFixed(2))}%`,
    },
  };

  return (
    <>
      <Dropdown menu={menuConfig} trigger={["click"]}>
        <a onClick={(e) => e.preventDefault()} style={{ color: "black" }}>
          <Space>
            {selectedItem || "No Available Characters"}
            <DownOutlined />
          </Space>
        </a>
      </Dropdown>
      <Modal
        title={"Create New Character"}
        open={isCreatingNewAgent}
        onCancel={closeNewAgentCreation}
        footer={null}
        maskClosable={false}
        closable={true}
        width={800}
      >
        <Form form={form} layout="vertical" onFinish={handleAgentCreation}>
          <Form.Item name="avatar" label="Upload Avatar">
            <ImgCrop rotationSlider>
              <Upload
                action="/api/upload_character_info"
                listType="picture-card"
                fileList={fileList}
                maxCount={1}
                onChange={onAvatarChange}
                onPreview={onAvatarPreview}
              >
                {fileList.length < 1 && "+ Upload"}
              </Upload>
            </ImgCrop>
          </Form.Item>
          <Form.Item
            name="name"
            label="AgentName"
            rules={[
              { required: true, message: "Please input the Character Name" },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="id"
            label="AgentID"
            rules={[
              { required: true, message: "Please input the Character ID" },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="description"
            label="AgentDescription"
            rules={[
              {
                required: true,
                message: "Please proivde the Character Description",
              },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="file"
            label="UploadFile"
            valuePropName="fileList"
            getValueFromEvent={(e) => e.fileList}
          >
            <Upload {...characterBackgroundProps}>
              <Button icon={<UploadOutlined />}>Upload Background</Button>
            </Upload>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">
              Create Character
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}

export default ModelSelector;
