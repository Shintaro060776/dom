FROM node:16 as build-stage

ENV OPENSSL_CONF=/dev/null

WORKDIR /app

COPY fluid/package.json ./
COPY fluid/yarn.lock ./

RUN yarn install

COPY fluid/ ./

RUN yarn build

FROM nginx:latest

COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY --from=build-stage /app/js /usr/share/nginx/html/js

CMD ["nginx", "-g", "daemon off;"]