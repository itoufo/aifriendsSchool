# Hosting Options

Deploy your knowledge framework to various hosting platforms.

## Overview

The framework builds to static files, making it compatible with many hosting options:
- 🌐 **Static hosting** services (Netlify, Vercel, GitHub Pages)
- ☁️ **Cloud platforms** (AWS S3, Google Cloud, Azure)
- 🔧 **Traditional hosting** (shared hosting, VPS)
- 🐳 **Container platforms** (Docker, Kubernetes)

## Static Hosting Platforms

### Netlify

**Advantages:**
- Automatic deployments from Git
- Built-in CDN and SSL
- Form handling and serverless functions
- Preview deployments for pull requests

**Setup:**
1. **Build configuration** (`netlify.toml`):
```toml
[build]
  publish = "dist"
  command = "npm run build"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[[headers]]
  for = "/assets/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"
```

2. **Deploy steps:**
```bash
# Connect your Git repository to Netlify
# Or deploy manually:
npm run build
npx netlify-cli deploy --prod --dir=dist
```

3. **Custom domain setup:**
```bash
# Add custom domain in Netlify dashboard
# DNS configuration:
# CNAME: your-domain.com -> your-site.netlify.app
```

### Vercel

**Advantages:**
- Excellent performance and edge caching
- Automatic preview deployments
- Built-in analytics
- Serverless function support

**Setup:**
1. **Configuration** (`vercel.json`):
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "installCommand": "npm ci",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
```

2. **Deploy:**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### GitHub Pages

**Advantages:**
- Free for public repositories
- Automatic deployments via GitHub Actions
- Easy setup for open-source projects

**Setup:**
1. **GitHub Actions workflow** (`.github/workflows/deploy.yml`):
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './dist'

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

2. **Configuration for subdirectory deployment:**
```typescript
// vite.config.ts - for GitHub Pages subdirectory
export default defineConfig({
  base: '/your-repository-name/', // Important for GitHub Pages
  // ... other config
});
```

## Cloud Hosting

### AWS S3 + CloudFront

**Advantages:**
- Highly scalable and reliable
- Global CDN with CloudFront
- Fine-grained access controls
- Cost-effective for large traffic

**Setup:**
1. **Create S3 bucket:**
```bash
# Using AWS CLI
aws s3 mb s3://your-knowledge-framework
aws s3 website s3://your-knowledge-framework --index-document index.html --error-document index.html
```

2. **Upload build:**
```bash
# Build and upload
npm run build
aws s3 sync dist/ s3://your-knowledge-framework --delete

# Set cache headers
aws s3 cp dist/assets/ s3://your-knowledge-framework/assets/ --recursive \
  --cache-control "public, max-age=31536000, immutable"
```

3. **CloudFront distribution:**
```json
{
  "DistributionConfig": {
    "Origins": [{
      "Id": "S3Origin",
      "DomainName": "your-knowledge-framework.s3.amazonaws.com",
      "S3OriginConfig": {
        "OriginAccessIdentity": ""
      }
    }],
    "DefaultCacheBehavior": {
      "TargetOriginId": "S3Origin",
      "ViewerProtocolPolicy": "redirect-to-https"
    },
    "CustomErrorResponses": [{
      "ErrorCode": 404,
      "ResponsePagePath": "/index.html",
      "ResponseCode": "200",
      "ErrorCachingMinTTL": 0
    }],
    "Enabled": true
  }
}
```

### Google Cloud Storage

**Setup:**
```bash
# Create bucket
gsutil mb gs://your-knowledge-framework

# Make bucket public
gsutil iam ch allUsers:objectViewer gs://your-knowledge-framework

# Upload files
npm run build
gsutil -m cp -r dist/* gs://your-knowledge-framework/

# Set up website
gsutil web set -m index.html -e index.html gs://your-knowledge-framework
```

### Azure Static Web Apps

**Setup:**
```yaml
# .github/workflows/azure-static-web-apps.yml
name: Azure Static Web Apps

on:
  push:
    branches: [ main ]

jobs:
  build_and_deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build And Deploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/"
          output_location: "dist"
          app_build_command: "npm run build"
```

## Traditional Hosting

### Shared Hosting / VPS

**Setup for cPanel/shared hosting:**
1. **Build locally:**
```bash
npm run build
```

2. **Upload via FTP/SFTP:**
```bash
# Using SCP
scp -r dist/* user@your-server.com:/public_html/

# Or use FileZilla, WinSCP, etc.
```

3. **Apache `.htaccess`** (for SPA routing):
```apache
# public_html/.htaccess
Options -MultiViews
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteRule ^ index.html [QR,L]

# Cache static assets
<IfModule mod_expires.c>
  ExpiresActive on
  ExpiresByType text/css "access plus 1 year"
  ExpiresByType application/javascript "access plus 1 year"
  ExpiresByType image/png "access plus 1 year"
  ExpiresByType image/jpg "access plus 1 year"
  ExpiresByType image/jpeg "access plus 1 year"
  ExpiresByType image/gif "access plus 1 year"
  ExpiresByType image/svg+xml "access plus 1 year"
</IfModule>
```

4. **Nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/knowledge-framework;
    index index.html;

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

## Container Deployment

### Docker

**Dockerfile:**
```dockerfile
# Build stage
FROM node:18-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine as production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**nginx.conf:**
```nginx
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    server {
        listen 80;
        server_name localhost;
        root /usr/share/nginx/html;
        index index.html;
        
        location / {
            try_files $uri $uri/ /index.html;
        }
        
        location /assets/ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

**Build and run:**
```bash
# Build image
docker build -t knowledge-framework .

# Run container
docker run -p 8080:80 knowledge-framework
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  knowledge-framework:
    build: .
    ports:
      - "8080:80"
    environment:
      - NODE_ENV=production
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    restart: unless-stopped
```

## CDN and Performance

### Cloudflare Setup

1. **Add your domain to Cloudflare**
2. **Configure caching rules:**
```javascript
// Cloudflare Workers script for advanced caching
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const cache = caches.default
  const url = new URL(request.url)
  
  // Cache static assets
  if (url.pathname.startsWith('/assets/')) {
    const cacheKey = new Request(url.toString(), request)
    let response = await cache.match(cacheKey)
    
    if (!response) {
      response = await fetch(request)
      const headers = new Headers(response.headers)
      headers.set('Cache-Control', 'public, max-age=31536000')
      
      response = new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: headers
      })
      
      event.waitUntil(cache.put(cacheKey, response.clone()))
    }
    
    return response
  }
  
  return fetch(request)
}
```

### Custom CDN Setup

**Using KeyCDN or similar:**
```bash
# Configure origin server
# Point CDN to your hosting provider
# Set cache rules:
# /assets/* : Cache for 1 year
# /*.html : Cache for 1 hour
# /* : Cache for 5 minutes
```

## Monitoring and Analytics

### Performance Monitoring
```html
<!-- Add to index.html -->
<script>
  // Basic performance monitoring
  window.addEventListener('load', () => {
    const navigation = performance.getEntriesByType('navigation')[0];
    const loadTime = navigation.loadEventEnd - navigation.fetchStart;
    
    // Send to your analytics service
    console.log('Page load time:', loadTime, 'ms');
  });
</script>
```

### Error Tracking
```typescript
// Error boundary for production
class ErrorBoundary extends React.Component {
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log to error tracking service
    console.error('Application error:', error, errorInfo);
  }
}
```

## Security Considerations

### Content Security Policy
```html
<!-- Add to index.html -->
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               script-src 'self' 'unsafe-inline'; 
               style-src 'self' 'unsafe-inline'; 
               img-src 'self' data: https:;">
```

### Security Headers
```nginx
# Add to nginx/apache configuration
add_header X-Frame-Options "SAMEORIGIN";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
```

## Deployment Automation

### Simple Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Deploying Knowledge Framework..."

# Build
echo "📦 Building..."
npm run build

# Upload to your hosting provider
echo "⬆️ Uploading..."
rsync -avz --delete dist/ user@server:/path/to/webroot/

# Clear CDN cache (if applicable)
echo "🗑️ Clearing cache..."
# curl -X POST "https://api.cloudflare.com/client/v4/zones/ZONE_ID/purge_cache" \
#   -H "Authorization: Bearer YOUR_TOKEN" \
#   -H "Content-Type: application/json" \
#   --data '{"purge_everything":true}'

echo "✅ Deployment complete!"
```

### Deployment Checklist

Before deploying:
- ✅ Run `npm run build` successfully
- ✅ Test with `npm run preview`
- ✅ Verify all content loads correctly
- ✅ Check responsive design
- ✅ Test quiz functionality
- ✅ Validate all internal links
- ✅ Ensure proper error pages (404)
- ✅ Test performance with Lighthouse
- ✅ Verify security headers
- ✅ Check analytics integration

## Troubleshooting

### Common Deployment Issues

**404 errors for routes:**
```nginx
# Missing SPA routing configuration
location / {
    try_files $uri $uri/ /index.html;
}
```

**Assets not loading:**
```typescript
// Incorrect base path in vite.config.ts
export default defineConfig({
  base: './', // Use relative paths
});
```

**MIME type issues:**
```nginx
# Missing MIME types
include /etc/nginx/mime.types;
```

**Performance issues:**
```bash
# Check bundle size
npm run build
ls -la dist/assets/

# Analyze with bundle analyzer
npm install --save-dev rollup-plugin-visualizer
```

## Cost Optimization

### Hosting Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| Netlify | 100GB bandwidth | $19+/month | Small to medium sites |
| Vercel | 100GB bandwidth | $20+/month | React applications |
| GitHub Pages | Free for public repos | N/A | Open source projects |
| AWS S3 + CloudFront | 12 months free | $1+/month | Large scale applications |
| Traditional hosting | Varies | $5+/month | Simple deployments |

### Bandwidth Optimization
```bash
# Compress assets
gzip -9 dist/assets/*.js
gzip -9 dist/assets/*.css

# Use WebP images
npm install --save-dev imagemin imagemin-webp
```

## Next Steps

- Set up automated deployments with CI/CD
- Implement monitoring and analytics
- Configure custom domains and SSL
- Optimize for search engines (SEO)
- Plan for scaling and performance optimization