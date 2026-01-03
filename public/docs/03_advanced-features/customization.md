# Customization

Learn how to customize the appearance, behavior, and features of your knowledge framework.

## Overview

The framework is designed to be highly customizable:
- 🎨 **Theme and styling** modifications
- ⚙️ **Feature configuration** options  
- 🔧 **Component customization** for advanced users
- 📱 **Responsive behavior** adjustments
- 🌐 **Multi-language support** setup

## Theme Customization

### CSS Variables
The easiest way to customize appearance is through CSS variables in `src/index.css`:

```css
:root {
  /* Primary colors */
  --primary-color: #2563eb;      /* Main accent color */
  --primary-hover: #1d4ed8;      /* Hover state */
  --primary-light: #dbeafe;      /* Light variant */
  
  /* Background colors */
  --bg-primary: #ffffff;         /* Main background */
  --bg-secondary: #f8fafc;       /* Secondary background */
  --sidebar-bg: #1e293b;         /* Sidebar background */
  --sidebar-hover: #334155;      /* Sidebar hover */
  
  /* Text colors */
  --text-primary: #1e293b;       /* Main text */
  --text-secondary: #64748b;     /* Secondary text */
  --text-muted: #94a3b8;         /* Muted text */
  --text-inverse: #ffffff;       /* Light text on dark bg */
  
  /* Interactive elements */
  --border-color: #e2e8f0;       /* Default borders */
  --success-color: #10b981;      /* Success states */
  --warning-color: #f59e0b;      /* Warning states */
  --error-color: #ef4444;        /* Error states */
  
  /* Layout */
  --sidebar-width: 280px;        /* Sidebar width */
  --content-max-width: 800px;    /* Content area max width */
  --border-radius: 8px;          /* Default border radius */
}
```

### Dark Mode Theme
```css
/* Dark mode variables */
[data-theme="dark"] {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --sidebar-bg: #020617;
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --border-color: #334155;
}

/* Dark mode toggle implementation */
.theme-toggle {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: 8px 12px;
  cursor: pointer;
}
```

### Custom Color Schemes
Create your own color palette:

```css
/* Blue theme */
.theme-blue {
  --primary-color: #3b82f6;
  --primary-hover: #2563eb;
  --primary-light: #dbeafe;
  --sidebar-bg: #1e40af;
}

/* Green theme */ 
.theme-green {
  --primary-color: #10b981;
  --primary-hover: #059669;
  --primary-light: #d1fae5;
  --sidebar-bg: #047857;
}

/* Purple theme */
.theme-purple {
  --primary-color: #8b5cf6;
  --primary-hover: #7c3aed;
  --primary-light: #ede9fe;
  --sidebar-bg: #6d28d9;
}
```

## Layout Customization

### Sidebar Configuration
Modify sidebar behavior in `src/components/Sidebar.tsx`:

```tsx
// Custom sidebar width
const SIDEBAR_WIDTH = '320px'; // Wider sidebar

// Collapsible sidebar
const [isCollapsed, setIsCollapsed] = useState(false);

const sidebarStyle = {
  width: isCollapsed ? '60px' : SIDEBAR_WIDTH,
  transition: 'width 0.3s ease'
};

// Auto-collapse on mobile
useEffect(() => {
  const handleResize = () => {
    setIsCollapsed(window.innerWidth < 768);
  };
  
  window.addEventListener('resize', handleResize);
  handleResize(); // Check on mount
  
  return () => window.removeEventListener('resize', handleResize);
}, []);
```

### Content Area Customization
Adjust content presentation in `src/components/MarkdownViewer.css`:

```css
/* Wider content area */
.markdown-viewer {
  max-width: 1000px; /* Increase from default 800px */
  margin: 0 auto;
  padding: 2rem;
}

/* Custom typography */
.markdown-viewer h1 {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  color: var(--primary-color);
}

.markdown-viewer h2 {
  font-size: 2rem;
  font-weight: 600;
  margin: 2rem 0 1rem 0;
  border-bottom: 2px solid var(--border-color);
  padding-bottom: 0.5rem;
}

/* Custom code blocks */
.markdown-viewer pre {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: 1rem;
  overflow-x: auto;
  font-family: 'Fira Code', 'Monaco', monospace;
}
```

## Feature Configuration

### Application Settings
Create a configuration file `src/config/appConfig.ts`:

```typescript
export interface AppConfig {
  // Quiz settings
  quiz: {
    passThreshold: number;        // 0.8 = 80%
    maxAttempts: number;          // 0 = unlimited
    showCorrectAnswers: boolean;  // Show answers after completion
    randomizeQuestions: boolean;  // Shuffle question order
    timeLimit?: number;           // Time limit in seconds
  };
  
  // Progress tracking
  progress: {
    enabled: boolean;             // Enable/disable progress tracking
    autoSave: boolean;           // Auto-save progress
    showInSidebar: boolean;      // Show progress indicators
    trackTimeSpent: boolean;     // Track time spent reading
  };
  
  // Notes and bookmarks
  notes: {
    enabled: boolean;            // Enable notes feature
    maxNoteLength: number;       // Maximum characters per note
    allowHighlighting: boolean;  // Enable text highlighting
    showInSidebar: boolean;     // Show bookmark panel
  };
  
  // UI preferences
  ui: {
    showBreadcrumbs: boolean;    // Show breadcrumb navigation
    enableSearch: boolean;       // Enable content search
    compactMode: boolean;        // Use compact layout
    animationsEnabled: boolean;  // Enable UI animations
  };
}

export const defaultConfig: AppConfig = {
  quiz: {
    passThreshold: 0.8,
    maxAttempts: 0,
    showCorrectAnswers: true,
    randomizeQuestions: false
  },
  progress: {
    enabled: true,
    autoSave: true,
    showInSidebar: true,
    trackTimeSpent: true
  },
  notes: {
    enabled: true,
    maxNoteLength: 1000,
    allowHighlighting: true,
    showInSidebar: true
  },
  ui: {
    showBreadcrumbs: true,
    enableSearch: false,
    compactMode: false,
    animationsEnabled: true
  }
};
```

### Using Configuration
```tsx
import { useConfig } from '../hooks/useConfig';

function QuizComponent() {
  const { config } = useConfig();
  
  // Use configuration values
  const canRetake = config.quiz.maxAttempts === 0 || 
                   attempts < config.quiz.maxAttempts;
  
  if (!config.quiz.showCorrectAnswers) {
    // Hide correct answers
  }
}
```

## Component Customization

### Custom Quiz Styling
Override quiz component styles:

```css
/* Custom quiz modal */
.quiz-modal {
  background: linear-gradient(135deg, var(--primary-color), var(--primary-hover));
  border-radius: 12px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.quiz-question {
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: var(--text-inverse);
}

.quiz-option {
  background: rgba(255, 255, 255, 0.1);
  border: 2px solid transparent;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.75rem;
  color: var(--text-inverse);
  cursor: pointer;
  transition: all 0.2s ease;
}

.quiz-option:hover {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
}

.quiz-option.selected {
  background: rgba(255, 255, 255, 0.9);
  color: var(--text-primary);
  border-color: white;
}
```

### Custom Progress Indicators
```css
/* Custom progress bar */
.progress-bar {
  background: var(--bg-secondary);
  border-radius: 100px;
  height: 8px;
  overflow: hidden;
  position: relative;
}

.progress-bar::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: linear-gradient(90deg, var(--primary-color), var(--success-color));
  width: var(--progress-percentage);
  transition: width 0.3s ease;
  border-radius: 100px;
}

/* Custom completion badges */
.chapter-status {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
}

.chapter-status.completed {
  background: var(--success-color);
  color: white;
}

.chapter-status.in-progress {
  background: var(--warning-color);
  color: white;
}

.chapter-status.not-started {
  background: var(--bg-secondary);
  border: 2px solid var(--border-color);
  color: var(--text-muted);
}
```

## Advanced Customization

### Custom Hooks
Create specialized hooks for your needs:

```typescript
// Custom analytics hook
export function useAnalytics() {
  const trackEvent = (event: string, properties?: Record<string, any>) => {
    // Custom analytics implementation
    console.log('Analytics event:', event, properties);
  };
  
  const trackPageView = (page: string) => {
    trackEvent('page_view', { page });
  };
  
  const trackQuizCompletion = (score: number, attempts: number) => {
    trackEvent('quiz_completed', { score, attempts });
  };
  
  return { trackEvent, trackPageView, trackQuizCompletion };
}

// Custom keyboard shortcuts hook
export function useKeyboardShortcuts() {
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'k':
            e.preventDefault();
            // Open search
            break;
          case 'b':
            e.preventDefault();
            // Toggle bookmark
            break;
          case 'n':
            e.preventDefault();
            // New note
            break;
        }
      }
    };
    
    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, []);
}
```

### Plugin System
Create a plugin architecture:

```typescript
// Plugin interface
interface Plugin {
  name: string;
  version: string;
  initialize: (app: AppContext) => void;
  cleanup?: () => void;
}

// Example plugin
const analyticsPlugin: Plugin = {
  name: 'analytics',
  version: '1.0.0',
  initialize: (app) => {
    // Add analytics tracking to quiz completions
    app.onQuizComplete((score) => {
      // Send to analytics service
    });
  }
};

// Plugin manager
class PluginManager {
  private plugins: Plugin[] = [];
  
  register(plugin: Plugin) {
    this.plugins.push(plugin);
    plugin.initialize(this.appContext);
  }
  
  unregister(pluginName: string) {
    const plugin = this.plugins.find(p => p.name === pluginName);
    if (plugin?.cleanup) {
      plugin.cleanup();
    }
    this.plugins = this.plugins.filter(p => p.name !== pluginName);
  }
}
```

## Responsive Design

### Mobile Optimizations
```css
/* Mobile-first responsive design */
@media (max-width: 768px) {
  .layout-container {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    height: auto;
    position: relative;
  }
  
  .content-area {
    padding: 1rem;
  }
  
  .markdown-viewer {
    font-size: 16px;
    line-height: 1.6;
  }
  
  .quiz-modal {
    margin: 1rem;
    max-width: calc(100vw - 2rem);
  }
}

/* Tablet adjustments */
@media (min-width: 769px) and (max-width: 1024px) {
  .sidebar {
    width: 240px;
  }
  
  .content-area {
    padding: 1.5rem;
  }
}
```

### Touch Interactions
```css
/* Touch-friendly interactive elements */
@media (hover: none) and (pointer: coarse) {
  .quiz-option,
  .sidebar-item,
  .note-item {
    min-height: 44px; /* Minimum touch target size */
    padding: 12px;
  }
  
  .quiz-option:active {
    background: rgba(255, 255, 255, 0.3);
    transform: scale(0.98);
  }
}
```

## Deployment Customization

### Environment-Specific Builds
```typescript
// src/config/environment.ts
export const environment = {
  production: process.env.NODE_ENV === 'production',
  development: process.env.NODE_ENV === 'development',
  apiUrl: process.env.VITE_API_URL || '',
  analytics: {
    enabled: process.env.VITE_ANALYTICS_ENABLED === 'true',
    trackingId: process.env.VITE_TRACKING_ID || ''
  }
};
```

### Custom Build Configuration
```javascript
// vite.config.ts customization
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'router': ['react-router-dom'],
          'markdown': ['react-markdown', 'remark-gfm']
        }
      }
    }
  },
  define: {
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __VERSION__: JSON.stringify(process.env.npm_package_version)
  }
});
```

## Best Practices

### Maintainability
- ✅ Use CSS variables for consistent theming
- ✅ Keep customizations in separate files when possible
- ✅ Document custom configurations
- ✅ Use TypeScript interfaces for configuration
- ✅ Test customizations across different devices

### Performance
- ✅ Minimize CSS bundle size
- ✅ Use CSS-in-JS sparingly for dynamic styles
- ✅ Optimize images and assets
- ✅ Lazy load heavy customizations
- ✅ Profile performance impact of customizations

### User Experience
- ✅ Maintain accessibility standards
- ✅ Provide fallbacks for custom features
- ✅ Test with different user preferences
- ✅ Consider color contrast ratios
- ✅ Ensure keyboard navigation works

## Next Steps

- Learn about [building for production](../04_deployment/building.md)
- Explore [hosting options](../04_deployment/hosting.md) for your customized framework
- Review the complete framework documentation for advanced features