import { useState, useEffect, useContext } from "react";
import { Button, Drawer, Form, Input, Flex, message } from "antd";
import axios from "axios";
import { StatusContext } from "../contexts/StatusContext";

const MyLogin = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [drawerVisible, setDrawerVisible] = useState(false);

  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;

  const verifyToken = async () => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        console.error("No token found");
        return;
      }

      const response = await axios.get("/api/auth/get_current_user", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.status === 200 && response.data) {
        setIsLoggedIn(true);
        setUsername(response.data.username);
        setCKStatus((prevState) => ({
          ...prevState,
          username: response.data.username,
        }));
        localStorage.setItem("username", response.data.username);
      } else {
        localStorage.setItem("username", "NA");
        console.error("Invalid token");
      }
    } catch (error) {
      console.error("Token verification failed:", error);
    }
  };
  const handleLogin = async (values: any) => {
    try {
      const formData = new FormData();
      formData.append("username", values.username);
      formData.append("password", values.password);

      const response = await axios.post("/api/auth/login", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (response.status === 200) {
        message.success("Login successful");
        setIsLoggedIn(true);
        setUsername(values.username);
        setCKStatus((prevState) => ({
          ...prevState,
          username: values.username,
        }));
        localStorage.setItem("token", response.data.access_token);
        localStorage.setItem("username", values.username);
        setDrawerVisible(false);
      } else if (response.status === 401) {
        message.error(
          "The username or password is incorrect. Please try again."
        );
      }
    } catch (error) {
      console.error("Login failed:", error);
    }
  };
  // 登录请求
  const handleRegister = async (values: any) => {
    if (values.password !== values.confirmPassword) {
      message.error("The passwords do not match. Please try again.");
      return;
    }
    if (values.username === "NA") {
      message.error("The username 'NA' is reserved. Please try another.");
      return;
    }
    try {
      // 向注册 API 发送请求
      const response = await axios.post("/api/auth/register", values);
      if (response.status === 201) {
        // 构造登录请求的数据
        message.success("Registration successful. ");
        // 调用登录接口
        setIsRegister(false);
        handleLogin(values);
        setDrawerVisible(false);
      } else if (response.status === 400) {
        // 用户名已存在
        message.error(
          "The username already exists. Please choose a different one."
        );
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response && error.response.status === 400) {
          // 用户名已存在
          message.error(
            "The username already exists. Please choose a different one."
          );
        } else {
          // 处理其他响应错误
          message.error("Registration failed. Please try again later.");
        }
      } else {
        // 处理非 Axios 错误
        console.error("An unexpected error occurred:", error);
        message.error("Registration failed. Please try again later.");
      }
      console.error("Registration failed:", error);
    }
  };

  const showLogin = () => {
    setIsRegister(false);
    setDrawerVisible(true);
  };

  const showRegister = () => {
    setIsRegister(true);
  };

  useEffect(() => {
    verifyToken();
  }, []);

  return (
    <div>
      {isLoggedIn ? (
        <Button
          type="dashed"
          block={true}
          size={"large"}
          onClick={() => setDrawerVisible(true)}
        >
          {username}
        </Button>
      ) : (
        <Button
          type="dashed"
          block={true}
          size={"large"}
          onClick={() => setDrawerVisible(true)}
        >
          登录
        </Button>
      )}
      <Drawer
        title={isRegister ? "Registration" : "Login"}
        placement="right"
        closable={false}
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
      >
        <Form onFinish={isRegister ? handleRegister : handleLogin}>
          <Form.Item
            name="username"
            rules={[{ required: true, message: "Please input the username" }]}
          >
            <Input placeholder="username" />
          </Form.Item>
          <Form.Item
            name="password"
            rules={[{ required: true, message: "Pleae input the password" }]}
          >
            <Input.Password placeholder="password" />
          </Form.Item>
          {isRegister && (
            <Form.Item
              name="confirmPassword"
              rules={[
                { required: true, message: "Please confirm the password" },
              ]}
            >
              <Input.Password placeholder="confirmPassword" />
            </Form.Item>
          )}
          <Form.Item>
            <Button type="dashed" block={true} size={"large"} htmlType="submit">
              {isRegister ? "Register" : "Login"}
            </Button>
          </Form.Item>
          <Button
            type="dashed"
            block={true}
            size={"large"}
            onClick={isRegister ? showLogin : showRegister}
          >
            {isRegister ? "Login" : "Register"}
          </Button>
        </Form>
      </Drawer>
    </div>
  );
};

export default MyLogin;
