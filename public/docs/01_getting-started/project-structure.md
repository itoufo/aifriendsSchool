# Project Structure

Understanding how the framework is organized will help you customize and extend it effectively.

## Directory Overview

```
markdown-knowledge-framework/
├── public/                     # Static assets
│   ├── docs/                  # 📝 Your Markdown content goes here
│   └── quizzes/               # 🎯 Quiz JSON files
├── src/
│   ├── components/            # 🔧 React UI components
│   ├── data/                  # 📊 Configuration and data
│   ├── hooks/                 # 🪝 Custom React hooks
│   ├── pages/                 # 📄 Main page components
│   ├── services/              # ⚙️ Business logic
│   └── App.tsx                # 🚪 Main application entry
├── package.json               # 📦 Dependencies and scripts
└── vite.config.ts            # ⚡ Vite configuration
```

## Key Files and Directories

### `/public/docs/` - Your Content
This is where all your Markdown files live. Organize them however makes sense for your content:

```
public/docs/
├── 01_getting-started/
│   ├── introduction.md
│   └── quick-start.md
├── 02_advanced-topics/
│   └── customization.md
└── images/
    └── diagram.png
```

### `/public/quizzes/` - Interactive Quizzes
JSON files that correspond to your documentation:

```
public/quizzes/
├── introduction.json          # Quiz for introduction.md
└── customization.json         # Quiz for customization.md
```

### `/src/data/curriculum.ts` - Navigation Structure
This file defines:
- How your content is organized in the sidebar
- Document metadata (titles, IDs, paths)
- Navigation order

```typescript
export const curriculum: Section[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    items: [
      { 
        id: 'introduction', 
        title: 'Introduction', 
        path: '/docs/01_getting-started/introduction.md' 
      }
    ]
  }
];
```

### `/src/components/` - UI Components
- **Layout.tsx** - Main application layout with sidebar
- **Sidebar.tsx** - Navigation sidebar with progress tracking
- **MarkdownViewer.tsx** - Renders Markdown content with syntax highlighting
- **QuizModal.tsx** - Interactive quiz overlay
- **QuizPlayer.tsx** - Quiz question interface

### `/src/hooks/` - Custom React Hooks
- **useProgress.ts** - Tracks learning progress and completion
- **useNotes.ts** - Manages user notes and bookmarks

### `/src/services/` - Business Logic  
- **quizService.ts** - Loads and manages quiz data
- **AppConfig.ts** - Application configuration settings

## Configuration Files

- **vite.config.ts** - Build tool configuration
- **tsconfig.json** - TypeScript compiler settings  
- **eslint.config.js** - Code linting rules
- **package.json** - Dependencies and npm scripts

## Customization Points

### Styling
- Global styles: `src/index.css`
- Component styles: Individual `.css` files next to components
- Theme colors: CSS variables in `src/index.css`

### Content Structure
- Modify `src/data/curriculum.ts` to change navigation
- Add new sections by creating directories in `public/docs/`
- Update document metadata in the curriculum configuration

### Features
- Add new hooks in `src/hooks/`
- Create custom components in `src/components/`
- Extend services in `src/services/`

## Next Steps

Now that you understand the structure, learn how to:
- [Create engaging content](../02_basic-usage/creating-content.md)
- [Set up navigation](../02_basic-usage/navigation-setup.md)
- [Add interactive quizzes](../02_basic-usage/quiz-system.md)