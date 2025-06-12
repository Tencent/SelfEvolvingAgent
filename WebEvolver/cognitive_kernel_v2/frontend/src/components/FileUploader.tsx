import { useContext } from "react";
import type { UploadProps } from "antd";
import { message, Upload, Button } from "antd";
import { StatusContext } from "../contexts/StatusContext";
import { UploadOutlined } from "@ant-design/icons";

function FileUploader() {
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;

  const props: UploadProps = {
    name: "file",
    multiple: true,
    customRequest(options) {
      const { file, onSuccess = () => {}, onError = () => {} } = options;

      const formData = new FormData();
      formData.append("file", file);

      formData.append("ckStatus", JSON.stringify(CKStatus));

      fetch("/api/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((result) => {
          console.log("Upload success:", result);
          onSuccess(result);
        })
        .catch((error) => {
          console.error("Upload error:", error);
          onError(error);
        });
    },
    onChange(info) {
      const { status } = info.file;
      if (status !== "uploading") {
        console.log(info.file, info.fileList);
      }
      if (status === "done") {
        message.success(`${info.file.name} file uploaded successfully.`);
        setCKStatus((prevStatus) => ({
          ...prevStatus,
          uploaded_files: [...prevStatus.uploaded_files, info.file.name],
        }));
      } else if (status === "error") {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    onDrop(e) {
      console.log("Dropped files", e.dataTransfer.files);
      const droppedFiles = Array.from(e.dataTransfer.files);
      const fileNames = droppedFiles.map((file) => file.name);
      setCKStatus((prevStatus) => ({
        ...prevStatus,
        uploaded_files: [...prevStatus.uploaded_files, ...fileNames],
      }));
    },
  };

  return (
    <Upload {...props}>
      <Button icon={<UploadOutlined />}></Button>
    </Upload>
  );
}

export default FileUploader;
