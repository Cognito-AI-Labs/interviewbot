# Base image
FROM python:3.13-slim
# Set workdir
WORKDIR /app
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install NGINX
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*
# Copy app files
COPY . .
# Copy NGINX config
COPY nginx.conf /etc/nginx/nginx.conf
# Expose ports
EXPOSE 7860 80
# Entrypoint: run NGINX and app
CMD ["sh", "-c", "service nginx start && python app.py"]

