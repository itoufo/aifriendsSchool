# Navigation Setup

Configure your sidebar navigation and content organization.

## Understanding the Curriculum Structure

The navigation is controlled by `src/data/curriculum.ts`. This file defines:

- **Sections**: Top-level categories in your sidebar
- **Items**: Individual documents within each section  
- **Metadata**: Titles, IDs, and file paths

## Basic Curriculum Configuration

```typescript
export const curriculum: Section[] = [
  {
    id: 'getting-started',           // Unique section identifier
    title: 'Getting Started',       // Display name in sidebar
    items: [
      {
        id: 'introduction',          // Unique document identifier  
        title: 'Introduction',      // Display name in sidebar
        path: '/docs/01_getting-started/introduction.md'  // File path
      },
      {
        id: 'quick-start',
        title: 'Quick Start Guide', 
        path: '/docs/01_getting-started/quick-start.md'
      }
    ]
  }
];
```

## Adding New Content

### Step 1: Create the Markdown File
```bash
# Create a new document
touch public/docs/02_basic-usage/new-topic.md
```

### Step 2: Add Content
```markdown
# New Topic

Your content goes here...
```

### Step 3: Register in Curriculum
```typescript
{
  id: 'basic-usage',
  title: 'Basic Usage',
  items: [
    { id: 'creating-content', title: 'Creating Content', path: '/docs/02_basic-usage/creating-content.md' },
    { id: 'new-topic', title: 'New Topic', path: '/docs/02_basic-usage/new-topic.md' }, // ← Add this line
  ]
}
```

## Navigation Features

### Automatic Progress Tracking
The sidebar automatically shows:
- ✅ Completed chapters (green checkmark)
- 📖 Currently reading (blue indicator)
- ⭕ Unread chapters (gray circle)

### Quick Navigation
- **Home button**: Returns to the main page
- **Section collapse**: Click section headers to collapse/expand
- **Progress indicators**: Visual feedback on completion status

## Advanced Configuration

### Custom Section Icons
Modify the sidebar component to add icons:

```tsx
// In src/components/Sidebar.tsx
const sectionIcons = {
  'getting-started': '🚀',
  'basic-usage': '📚',
  'advanced-features': '⚙️',
  'deployment': '🌐'
};
```

### Nested Subsections
For complex hierarchies, you can create nested structures:

```typescript
{
  id: 'advanced-guides',
  title: 'Advanced Guides',
  items: [
    { id: 'api-reference', title: 'API Reference', path: '/docs/advanced/api.md' },
    { id: 'customization', title: 'Customization Guide', path: '/docs/advanced/customization.md' },
    // Group related topics
    { id: 'integrations', title: 'Third-party Integrations', path: '/docs/advanced/integrations.md' },
  ]
}
```

### Document Ordering
The order in `curriculum.ts` determines the sidebar order and navigation flow:

```typescript
// Documents appear in this order in the sidebar
items: [
  { id: 'intro', title: '1. Introduction', path: '/docs/intro.md' },      // First
  { id: 'setup', title: '2. Setup', path: '/docs/setup.md' },            // Second  
  { id: 'usage', title: '3. Basic Usage', path: '/docs/usage.md' }       // Third
]
```

## Navigation Utilities

The framework provides helper functions for navigation:

### `getDocById(id: string)`
```typescript
// Get document metadata by ID
const doc = getDocById('introduction');
console.log(doc?.title); // "Introduction"
```

### `getDocByPath(path: string)`
```typescript
// Get document metadata by file path
const doc = getDocByPath('/docs/01_getting-started/introduction.md');
```

### `getNextDoc(currentId: string)`
```typescript
// Get the next document in sequence
const nextDoc = getNextDoc('introduction');
// Returns: { id: 'quick-start', title: 'Quick Start Guide', ... }
```

## URL Structure

The framework uses clean URLs that match your content structure:

```
/                                    → Home page
/doc/introduction                    → Introduction document  
/doc/quick-start                     → Quick Start document
/doc/advanced-features               → Advanced features document
```

## Best Practices

### Naming Conventions
- **IDs**: Use kebab-case: `user-authentication`, `api-reference`
- **Titles**: Use proper case: "User Authentication", "API Reference"  
- **Paths**: Match your file structure: `/docs/section/document.md`

### Organization Tips
1. **Group related content** in sections
2. **Order logically** from basic to advanced
3. **Use descriptive titles** that clearly indicate content
4. **Keep sections balanced** (3-7 items per section works well)

### File Path Management
- Always use absolute paths starting with `/docs/`
- Match your actual file structure in `public/docs/`
- Use consistent directory naming (numbers + descriptive names)

## Troubleshooting

### Content Not Appearing
1. Check file path in `curriculum.ts` matches actual file location
2. Ensure ID is unique across all documents
3. Verify Markdown file exists in `public/docs/`
4. Restart dev server if changes don't appear

### Navigation Issues  
1. Check for duplicate IDs in curriculum
2. Ensure proper TypeScript syntax in curriculum.ts
3. Verify all required fields (id, title, path) are present

## Next Steps

- Set up [interactive quizzes](quiz-system.md) for your content
- Learn about [progress tracking](../03_advanced-features/progress-tracking.md)
- Explore [customization options](../03_advanced-features/customization.md)