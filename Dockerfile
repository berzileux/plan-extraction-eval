FROM nginxinc/nginx-unprivileged:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY viewer/ /usr/share/nginx/html/
EXPOSE 8080
