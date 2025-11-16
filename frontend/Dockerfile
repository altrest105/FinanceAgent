FROM node:18

WORKDIR /app

COPY package.json package-lock.json /app/

RUN npm install

COPY . /app/

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
