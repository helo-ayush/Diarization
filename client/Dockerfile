FROM node:20-alpine

WORKDIR /app

# Install dependencies first for Docker caching
COPY package.json package-lock.json* ./
RUN npm install

# Copy all local files
COPY . .

# Expose Vite's default port
EXPOSE 5173

# Run the dev server and expose it to Docker host network
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
