# 构建阶段
FROM node:latest as build

# 设置工作目录
WORKDIR /app

# 复制 package.json 和 package-lock.json（如果存在）
COPY package*.json ./

# 安装项目依赖
RUN npm install

RUN npx playwright install
RUN npx playwright install-deps
RUN npm install uuid
RUN npm install js-yaml
RUN npm install playwright-extra puppeteer-extra-plugin-stealth
# RUN npm install @axe-core/playwright



# 复制项目文件到工作目录
COPY . .

CMD ["npm", "start"]
