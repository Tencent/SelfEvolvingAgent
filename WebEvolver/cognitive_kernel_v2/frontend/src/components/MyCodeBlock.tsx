import React, { useState } from "react";
import { CopyOutlined } from "@ant-design/icons";
import { CodeBlock, atomOneLight } from "react-code-blocks";

interface CodeBlockProps {
  language: string;
  code: string;
}

const MyCodeBlock: React.FC<CodeBlockProps> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);
  const codeLines = code.split("\n");

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span>{language}</span>
        <button onClick={handleCopy} style={styles.iconButton}>
          <CopyOutlined />
        </button>
      </div>
      <CodeBlock
        text={code}
        language={language}
        showLineNumbers={true}
        theme={atomOneLight}
        // wrapLines
      />
    </div>
  );
};

const styles = {
  container: {
    backgroundColor: "#f5f5f5",
    borderRadius: "5px",
    padding: "10px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
    margin: "20px 0",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
    backgroundColor: "transparent",
  },
  button: {
    border: "none",
    backgroundColor: "#007bff",
    color: "white",
    padding: "5px 10px",
    borderRadius: "5px",
    cursor: "pointer",
  },
  codeBlock: {
    backgroundColor: "#fff",
    padding: "10px",
    overflowX: "auto" as "auto",
    borderRadius: "5px",
  },
  codeLine: {
    display: "flex",
    alignItems: "flex-start",
    lineHeight: "1.5",
    fontFamily: "monospace",
  },
  lineNumber: {
    color: "#888",
    marginRight: "10px",
    minWidth: "20px",
    userSelect: "none" as "none",
  },
  iconButton: {
    border: "none",
    backgroundColor: "transparent",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};

export default MyCodeBlock;
