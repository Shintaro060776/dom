FROM nginx:latest

COPY fluid/ /app/

RUN rm -rf /usr/share/nginx/html && ln -s /app/dist /usr/share/nginx/html

CMD ["nginx", "-g", "daemon off;"]