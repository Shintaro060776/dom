FROM node:16 as build-stage

ENV OPENSSL_CONF=/dev/null

WORKDIR /app

COPY /home/runner/work/next/next/fluid/package.json /home/runner/work/next/next/fluid/yarn.lock ./

RUN yarn install

COPY /home/runner/work/next/next/fluid/ ./

RUN yarn build

FROM nginx:latest

COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY --from=build-stage /app/js /usr/share/nginx/html/js

CMD ["nginx", "-g", "daemon off;"]