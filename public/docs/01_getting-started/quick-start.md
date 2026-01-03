# Quick Start Guide

Get your knowledge framework up and running in minutes!

## Prerequisites

- Node.js (18+ recommended)
- npm or yarn
- Basic knowledge of Markdown

## Installation

1. **Clone or download this framework**
```bash
git clone <your-repo-url>
cd markdown-knowledge-framework
```

2. **Install dependencies**
```bash
npm install
```

3. **Start the development server**
```bash
npm run dev
```

4. **Open your browser**
Navigate to `http://localhost:5173` to see your knowledge framework in action!

## Adding Your First Content

1. **Create a new Markdown file**
```bash
touch public/docs/01_getting-started/my-first-doc.md
```

2. **Add some content**
```markdown
# My First Document

This is my first piece of content in the knowledge framework!

## Features I love:
- Easy Markdown editing
- Automatic navigation
- Built-in progress tracking
```

3. **Register it in the curriculum**
Edit `src/data/curriculum.ts` and add your new document:
```typescript
{
  id: 'my-first-doc',
  title: 'My First Document', 
  path: '/docs/01_getting-started/my-first-doc.md'
}
```

4. **Refresh your browser** - your new content appears automatically!

## Next Steps

- Learn about [Project Structure](project-structure.md) to understand the framework layout
- Check out [Creating Content](../02_basic-usage/creating-content.md) for advanced Markdown tips
- Set up [Quiz System](../02_basic-usage/quiz-system.md) to add interactive elements

## Development Commands

```bash
npm run dev      # Start development server
npm run build    # Build for production  
npm run preview  # Preview production build
npm run lint     # Run linting
```

That's it! You're ready to start building your knowledge base. 🚀