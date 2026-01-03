# Creating Content

Learn how to create engaging Markdown content for your knowledge framework.

## Markdown Basics

The framework supports full GitHub Flavored Markdown (GFM) with additional features.

### Text Formatting

```markdown
# Heading 1
## Heading 2  
### Heading 3

**Bold text**
*Italic text*
~~Strikethrough~~
`Inline code`
```

### Lists and Links

```markdown
- Bullet point 1
- Bullet point 2
  - Nested item

1. Numbered list
2. Another item

[Link text](https://example.com)
[Internal link](../01_getting-started/introduction.md)
```

### Code Blocks

````markdown
```javascript
function greet(name) {
  console.log(`Hello, ${name}!`);
}
```
````

### Tables

```markdown
| Feature | Supported |
|---------|-----------|
| Tables  | ✅ Yes    |
| Math    | ✅ Yes    |
| Diagrams| ⭕ Coming |
```

## Advanced Features

### Callout Boxes

Use standard Markdown blockquotes for callouts:

```markdown
> **💡 Tip**: This is a helpful tip for readers.

> **⚠️ Warning**: Important information to be aware of.

> **📝 Note**: Additional context or explanation.
```

### Images and Media

```markdown
![Alt text](./images/diagram.png)
![External image](https://example.com/image.jpg)
```

Place images in `public/docs/images/` or alongside your Markdown files.

## Content Organization Tips

### File Naming
- Use clear, descriptive names: `advanced-configuration.md`
- Include numbers for ordering: `01_introduction.md`, `02_setup.md`  
- Use hyphens, not spaces: `user-guide.md` ✅, `user guide.md` ❌

### Directory Structure
```
docs/
├── 01_getting-started/
│   ├── introduction.md
│   └── images/
│       └── welcome.png
├── 02_user-guide/
│   ├── basic-usage.md
│   └── advanced-features.md
```

### Document Structure
Each document should have:

1. **Clear heading** (H1) as the title
2. **Brief introduction** explaining what readers will learn
3. **Logical sections** with H2/H3 headings
4. **Code examples** where relevant
5. **Next steps** linking to related content

## Linking Between Documents

### Relative Links
```markdown
[Previous: Introduction](../01_getting-started/introduction.md)
[Next: Navigation Setup](./navigation-setup.md)
```

### Cross-References
```markdown
See the [Project Structure](../01_getting-started/project-structure.md#key-files-and-directories) 
for more details on organizing files.
```

## Content Best Practices

### Writing Style
- ✅ Use clear, concise language
- ✅ Include practical examples
- ✅ Break up long sections with subheadings
- ✅ Add visual elements (images, code blocks, tables)
- ❌ Avoid overly technical jargon
- ❌ Don't make assumptions about prior knowledge

### Accessibility
- Add alt text to images
- Use descriptive link text (not "click here")
- Ensure good heading hierarchy (don't skip levels)
- Include code comments in examples

### Performance
- Optimize images (WebP format recommended)
- Keep file sizes reasonable (<2MB per document)
- Use external links sparingly

## Testing Your Content

1. **Preview locally**: `npm run dev` and check your changes
2. **Test all links**: Ensure internal links work correctly  
3. **Check mobile view**: Verify content looks good on small screens
4. **Validate Markdown**: Use tools like markdownlint

## Next Steps

- Learn how to [set up navigation](navigation-setup.md) for your content
- Add [interactive quizzes](quiz-system.md) to test understanding
- Explore [advanced features](../03_advanced-features/progress-tracking.md) like progress tracking