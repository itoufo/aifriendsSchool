# Markdown Knowledge Framework

A fast, lightweight, and extensible framework for creating knowledge bases, documentation sites, and educational content using Markdown files.

## ✨ Features

- 📝 **Markdown-driven content** - Write in simple Markdown, deploy as a fast web app
- 🚀 **Zero database required** - Everything is file-based for easy maintenance  
- 📊 **Built-in progress tracking** - Automatic learning progress with quiz completion
- 🎯 **Interactive quizzes** - JSON-based quiz system with immediate feedback
- 📝 **Notes & bookmarks** - Let users take notes and bookmark important sections
- ⚡ **Fast performance** - Static generation with Vite for optimal loading speeds
- 📱 **Responsive design** - Works perfectly on desktop and mobile devices
- 🎨 **Customizable themes** - Easy styling with CSS variables
- 🔧 **Extensible architecture** - Add custom features with React hooks and components

## 🚀 Quick Start

```bash
# Clone the framework
git clone <this-repository>
cd markdown-knowledge-framework

# Install dependencies
npm install

# Start development server
npm run dev

# Open your browser to http://localhost:5173
```

## 📁 Project Structure

```
├── public/
│   ├── docs/                    # 📝 Your Markdown content
│   │   ├── 01_getting-started/
│   │   ├── 02_basic-usage/
│   │   └── 03_advanced-features/
│   └── quizzes/                 # 🎯 Quiz JSON files
├── src/
│   ├── components/              # 🔧 React UI components  
│   ├── data/curriculum.ts       # 📊 Content navigation structure
│   ├── hooks/                   # 🪝 Custom React hooks
│   └── services/                # ⚙️ Business logic
└── package.json                 # 📦 Dependencies and scripts
```

## 📖 Adding Your Content

### 1. Create Markdown Files
Add your content to `public/docs/`:

```markdown
# My New Topic

Your content here using standard Markdown syntax.

## Key Features
- Easy to write
- Fast to deploy  
- Built-in search and navigation
```

### 2. Register in Navigation
Add to `src/data/curriculum.ts`:

```typescript
{
  id: 'my-new-topic',
  title: 'My New Topic',
  path: '/docs/01_getting-started/my-new-topic.md'
}
```

### 3. Add Quiz (Optional)
Create `public/quizzes/my-new-topic.json`:

```json
{
  "docId": "my-new-topic",
  "title": "Topic Quiz",
  "questions": [
    {
      "question": "What makes this framework special?",
      "options": ["Speed", "Simplicity", "Features", "All of the above"],
      "correctAnswer": 3,
      "explanation": "The framework combines speed, simplicity, and rich features."
    }
  ]
}
```

## 🛠️ Development Commands

```bash
npm run dev      # Start development server
npm run build    # Build for production  
npm run preview  # Preview production build
npm run lint     # Run code linting
```

## 🎨 Customization

### Theme Colors
Edit CSS variables in `src/index.css`:

```css
:root {
  --primary-color: #2563eb;      /* Main accent color */
  --sidebar-bg: #1e293b;         /* Sidebar background */  
  --text-primary: #1e293b;       /* Main text color */
}
```

### Layout Options  
Modify `src/components/Layout.tsx` for custom layouts, sidebar behavior, or responsive breakpoints.

### Feature Configuration
Customize quiz behavior, progress tracking, and notes in `src/services/` directory.

## 📚 Built-in Features

### Progress Tracking
- Automatic chapter completion based on quiz scores (80% pass rate)
- Time tracking for each section
- Visual progress indicators in sidebar
- Persistent storage in browser localStorage

### Quiz System  
- Multiple choice questions with explanations
- Immediate feedback and scoring
- Retake capability for improvement
- JSON-based for easy content management

### Notes & Bookmarks
- Chapter-specific note taking
- Quick bookmark system  
- Text highlighting with persistent storage
- Tag-based organization
- Export/import functionality

## 🚀 Deployment

The framework builds to static files compatible with any hosting service:

### Quick Deploy
```bash
npm run build
# Upload dist/ folder to your hosting provider
```

### Popular Platforms
- **Netlify**: Drag & drop `dist` folder or connect Git repo
- **Vercel**: `vercel --prod` after building  
- **GitHub Pages**: Use provided GitHub Actions workflow
- **AWS S3**: Upload to bucket with static website hosting
- **Traditional hosting**: Upload via FTP to your web server

## 📖 Documentation

Comprehensive guides available in the framework:

- **[Getting Started](public/docs/01_getting-started/)** - Setup and basic usage
- **[Content Creation](public/docs/02_basic-usage/)** - Writing and organizing content  
- **[Advanced Features](public/docs/03_advanced-features/)** - Customization and extensibility
- **[Deployment](public/docs/04_deployment/)** - Hosting and production setup

## 🔧 Technical Stack

- **React 19** - Modern UI framework
- **TypeScript** - Type safety and developer experience
- **Vite** - Fast build tool and dev server  
- **React Router** - Client-side routing
- **React Markdown** - Markdown rendering with syntax highlighting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add new feature"`
5. Push and create a pull request

## 📄 License

MIT License - feel free to use this framework for any project!

## 🆘 Support

- 📚 **Documentation**: Check the built-in guides in `public/docs/`
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💡 **Feature Requests**: Suggest improvements via GitHub Issues
- 📧 **Questions**: Use GitHub Discussions for general questions

## 🏗️ Use Cases

Perfect for:
- **Documentation sites** for software projects
- **Educational content** and online courses  
- **Knowledge bases** for teams and organizations
- **Training materials** with interactive elements
- **Technical guides** with progress tracking
- **Learning platforms** with quiz integration

## ⭐ Why Choose This Framework?

- **No vendor lock-in** - Standard web technologies
- **Easy migration** - Content in portable Markdown format
- **Low maintenance** - File-based, no database to manage  
- **Fast performance** - Static generation for optimal speed
- **Rich features** - Progress tracking, quizzes, notes out of the box
- **Extensible** - Add custom features with standard React patterns

---

**Ready to build your knowledge base?** Start with `npm run dev` and explore the included documentation! 🚀