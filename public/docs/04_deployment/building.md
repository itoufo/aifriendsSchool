# Building for Production

Learn how to build and optimize your knowledge framework for production deployment.

## Build Process Overview

The framework uses Vite for fast, optimized production builds:
- 📦 **Asset bundling** with automatic optimization
- 🗜️ **Code minification** for smaller file sizes
- 📱 **Tree shaking** to remove unused code
- 🖼️ **Asset optimization** for images and fonts
- 🔗 **Chunk splitting** for better caching

## Basic Build Commands

### Development Build
```bash
# Start development server with hot reload
npm run dev
```

### Production Build
```bash
# Create optimized production build
npm run build

# Preview production build locally
npm run preview
```

### Linting and Type Checking
```bash
# Run ESLint for code quality
npm run lint

# Type check with TypeScript (if configured)
npx tsc --noEmit
```

## Build Output Structure

After running `npm run build`, you'll find these files in the `dist/` directory:

```
dist/
├── index.html              # Main HTML file
├── assets/
│   ├── index-[hash].js     # Main JavaScript bundle
│   ├── index-[hash].css    # Compiled CSS
│   └── logo-[hash].svg     # Optimized assets
├── docs/                   # Your Markdown content
│   ├── 01_getting-started/
│   └── 02_basic-usage/
├── quizzes/               # Quiz JSON files
│   ├── introduction.json
│   └── quick-start.json
└── vite.svg              # Static assets
```

## Build Optimization

### Vite Configuration
Customize your build in `vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    // Output directory (default: dist)
    outDir: 'dist',
    
    // Generate source maps for debugging
    sourcemap: true,
    
    // Minify code
    minify: 'terser',
    
    // Chunk size warning limit (default: 500kb)
    chunkSizeWarningLimit: 1000,
    
    rollupOptions: {
      output: {
        // Manual chunk splitting for better caching
        manualChunks: {
          // Separate vendor libraries
          'react-vendor': ['react', 'react-dom'],
          'router': ['react-router-dom'],
          'markdown': ['react-markdown', 'remark-gfm'],
        },
        
        // Custom file naming
        chunkFileNames: 'js/[name]-[hash].js',
        entryFileNames: 'js/[name]-[hash].js',
        assetFileNames: ({ name }) => {
          if (/\.(gif|jpe?g|png|svg)$/.test(name ?? '')) {
            return 'images/[name]-[hash][extname]';
          }
          if (/\.css$/.test(name ?? '')) {
            return 'css/[name]-[hash][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        }
      }
    }
  },
  
  // Base URL for deployment (important for subdirectory deployments)
  base: './', // Use relative paths
  
  // Define environment variables
  define: {
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0')
  }
});
```

### Advanced Optimization
```typescript
// Additional optimization options
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Separate node_modules into vendor chunks
          if (id.includes('node_modules')) {
            if (id.includes('react')) {
              return 'react-vendor';
            }
            if (id.includes('markdown') || id.includes('remark')) {
              return 'markdown-vendor';
            }
            return 'vendor';
          }
          
          // Separate large components
          if (id.includes('components/QuizPlayer')) {
            return 'quiz-components';
          }
        }
      }
    },
    
    // Terser configuration for better compression
    terserOptions: {
      compress: {
        drop_console: true,  // Remove console.log in production
        drop_debugger: true, // Remove debugger statements
      }
    }
  }
});
```

## Asset Optimization

### Image Optimization
```bash
# Install imagemin for image optimization
npm install --save-dev vite-plugin-imagemin imagemin-webp

# Add to vite.config.ts
import { imageOptimize } from 'vite-plugin-imagemin';

export default defineConfig({
  plugins: [
    react(),
    imageOptimize({
      gifsicle: { optimizationLevel: 7 },
      mozjpeg: { quality: 85 },
      pngquant: { quality: [0.8, 0.9] },
      svgo: {
        plugins: [
          { name: 'removeViewBox', active: false },
          { name: 'removeEmptyAttrs', active: true }
        ]
      },
      webp: { quality: 85 }
    })
  ]
});
```

### Font Optimization
```css
/* Optimize web font loading */
@font-face {
  font-family: 'CustomFont';
  src: url('./fonts/custom-font.woff2') format('woff2'),
       url('./fonts/custom-font.woff') format('woff');
  font-display: swap; /* Improves loading performance */
  font-weight: 400;
  font-style: normal;
}

/* Preload critical fonts */
/* Add to index.html */
<link rel="preload" href="/fonts/main-font.woff2" as="font" type="font/woff2" crossorigin>
```

## Performance Optimization

### Code Splitting
```typescript
// Lazy load heavy components
const QuizModal = lazy(() => import('./components/QuizModal'));
const NotesPanel = lazy(() => import('./components/NotesPanel'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/doc/:docId" element={<DocPage />} />
        </Routes>
      </Router>
    </Suspense>
  );
}
```

### Bundle Analysis
```bash
# Install bundle analyzer
npm install --save-dev rollup-plugin-visualizer

# Add to vite.config.ts
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: 'bundle-analysis.html',
      open: true,
      gzipSize: true,
      brotliSize: true
    })
  ]
});

# Build and analyze
npm run build
# Opens bundle-analysis.html in browser
```

## Environment Configuration

### Environment Variables
Create environment files:

```bash
# .env.local (local development)
VITE_APP_TITLE=My Knowledge Framework
VITE_ANALYTICS_ID=your-analytics-id
VITE_API_URL=https://api.example.com

# .env.production (production build)
VITE_APP_TITLE=Knowledge Framework
VITE_ANALYTICS_ID=prod-analytics-id
VITE_API_URL=https://api.production.com
```

Use in your code:
```typescript
// src/config/environment.ts
export const config = {
  appTitle: import.meta.env.VITE_APP_TITLE || 'Knowledge Framework',
  analyticsId: import.meta.env.VITE_ANALYTICS_ID,
  apiUrl: import.meta.env.VITE_API_URL,
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD
};
```

### Build Scripts
Add custom build scripts to `package.json`:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "build:analyze": "npm run build && npx vite-bundle-analyzer dist/stats.html",
    "build:staging": "NODE_ENV=staging vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "type-check": "tsc --noEmit",
    "clean": "rm -rf dist",
    "prebuild": "npm run clean && npm run lint && npm run type-check"
  }
}
```

## Content Optimization

### Markdown Processing
Optimize Markdown content during build:

```typescript
// Build script to process Markdown files
import fs from 'fs';
import path from 'path';

const optimizeMarkdown = () => {
  const docsDir = path.join(process.cwd(), 'public/docs');
  
  // Process all .md files
  const processDirectory = (dir: string) => {
    const files = fs.readdirSync(dir);
    
    files.forEach(file => {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory()) {
        processDirectory(filePath);
      } else if (file.endsWith('.md')) {
        let content = fs.readFileSync(filePath, 'utf8');
        
        // Optimize images paths for production
        content = content.replace(
          /!\[([^\]]*)\]\(\.\/images\/([^)]+)\)/g,
          '![$1](/docs/images/$2)'
        );
        
        // Remove comments
        content = content.replace(/<!--[\s\S]*?-->/g, '');
        
        // Normalize line endings
        content = content.replace(/\r\n/g, '\n');
        
        fs.writeFileSync(filePath, content);
      }
    });
  };
  
  processDirectory(docsDir);
};

// Run during build
optimizeMarkdown();
```

## Testing Build

### Automated Testing
```bash
# Test build process
npm run build

# Verify build output
ls -la dist/

# Check for missing files
test -f dist/index.html && echo "✅ index.html exists"
test -d dist/docs && echo "✅ docs directory exists"
test -d dist/quizzes && echo "✅ quizzes directory exists"

# Test with production server
npm run preview
```

### Build Validation Script
```bash
#!/bin/bash
# build-check.sh

echo "🔍 Validating build..."

# Check if build succeeded
if [ ! -d "dist" ]; then
  echo "❌ Build failed: dist directory not found"
  exit 1
fi

# Check critical files
critical_files=("index.html" "docs" "quizzes")
for file in "${critical_files[@]}"; do
  if [ ! -e "dist/$file" ]; then
    echo "❌ Missing critical file: $file"
    exit 1
  fi
done

# Check bundle size
bundle_size=$(du -sh dist | cut -f1)
echo "📦 Bundle size: $bundle_size"

# Check for source maps (if enabled)
if ls dist/assets/*.map 1> /dev/null 2>&1; then
  echo "🗺️ Source maps generated"
fi

echo "✅ Build validation passed!"
```

## Common Build Issues

### TypeScript Errors
```bash
# Fix TypeScript issues
npm run type-check

# Common fixes:
# 1. Add type definitions
npm install --save-dev @types/node

# 2. Update tsconfig.json
{
  "compilerOptions": {
    "skipLibCheck": true,  // Skip type checking of declaration files
    "allowSyntheticDefaultImports": true
  }
}
```

### Memory Issues
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
npm run build

# Or add to package.json scripts
"build": "NODE_OPTIONS='--max-old-space-size=4096' vite build"
```

### Path Issues
```typescript
// Fix asset path issues in vite.config.ts
export default defineConfig({
  base: './',  // Use relative paths
  publicDir: 'public',
  
  build: {
    assetsDir: 'assets',
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html')
      }
    }
  }
});
```

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/build.yml
name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run linting
      run: npm run lint
    
    - name: Run type check
      run: npm run type-check
    
    - name: Build
      run: npm run build
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
```

## Next Steps

- Learn about [hosting options](hosting.md) for your built application
- Explore different deployment strategies and platforms
- Set up continuous deployment for automated builds