# 构建阶段
FROM node:latest as build

# 设置工作目录
WORKDIR /app

# 复制package.json和package-lock.json（如果存在）
COPY package*.json ./

# 安装项目依赖，包括开发依赖
RUN npm install
RUN npm install antd-img-crop --save
RUN npm install react-json-view --legacy-peer-deps



# 复制项目文件到工作目录
COPY . .

# 使用Vite构建项目
RUN npm run build

# 部署阶段
FROM nginx:alpine

# 安装bash
RUN apk add --no-cache bash

# 从构建阶段复制构建产物到nginx目录
COPY --from=build /app/dist /usr/share/nginx/html

# 如果你的应用需要重定向，可以添加自定义nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 暴露80端口
EXPOSE 80

# 启动nginx，无需额外命令，因为使用了nginx基础镜像的默认CMD
