# Quiz System

Add interactive quizzes to test understanding and track learning progress.

## Overview

The quiz system provides:
- 📝 **Multiple-choice questions** with explanations
- 🎯 **Progress tracking** based on quiz completion  
- 🏆 **Pass/fail scoring** (80% threshold by default)
- 💾 **Persistent results** stored in browser localStorage
- 🔄 **Retake capability** for improvement

## Creating a Quiz

### Step 1: Create the Quiz File

Create a JSON file in `public/quizzes/` that matches your document ID:

```bash
# For a document with id 'introduction'
touch public/quizzes/introduction.json
```

### Step 2: Define Quiz Structure

```json
{
  "docId": "introduction",
  "title": "Introduction Quiz", 
  "description": "Test your understanding of the framework basics",
  "questions": [
    {
      "id": "q1",
      "question": "What is the primary format for content in this framework?",
      "options": [
        "HTML files",
        "Markdown files", 
        "JSON files",
        "XML files"
      ],
      "correctAnswer": 1,
      "explanation": "The framework is built around Markdown files for easy content creation and management."
    },
    {
      "id": "q2", 
      "question": "Which file controls the navigation structure?",
      "options": [
        "package.json",
        "App.tsx",
        "curriculum.ts",
        "vite.config.ts"
      ],
      "correctAnswer": 2,
      "explanation": "The curriculum.ts file in src/data/ defines all navigation and document metadata."
    }
  ]
}
```

### Step 3: Link to Document

Quizzes are automatically linked to documents by matching the quiz filename to the document ID:

```typescript
// In src/data/curriculum.ts
{ 
  id: 'introduction',        // ← This matches introduction.json
  title: 'Introduction', 
  path: '/docs/intro.md' 
}
```

## Quiz JSON Schema

### Root Properties
```json
{
  "docId": "string",          // Must match document ID in curriculum
  "title": "string",          // Quiz title shown to users  
  "description": "string",    // Optional description
  "questions": []             // Array of question objects
}
```

### Question Properties
```json
{
  "id": "string",            // Unique question identifier
  "question": "string",      // The question text
  "options": ["string"],     // Array of 2-6 answer choices
  "correctAnswer": "number", // Index of correct option (0-based)
  "explanation": "string"    // Optional explanation shown after answering
}
```

## Question Types and Examples

### Basic Multiple Choice
```json
{
  "id": "basic-1",
  "question": "What build tool does this framework use?",
  "options": ["Webpack", "Vite", "Parcel", "Rollup"],
  "correctAnswer": 1,
  "explanation": "Vite is used for fast development and optimized production builds."
}
```

### True/False Questions
```json
{
  "id": "tf-1", 
  "question": "The framework requires a database to store content.",
  "options": ["True", "False"],
  "correctAnswer": 1,
  "explanation": "The framework is completely file-based and requires no database."
}
```

### Code-Related Questions
```json
{
  "id": "code-1",
  "question": "Which command starts the development server?",
  "options": [
    "npm start",
    "npm run dev", 
    "npm run serve",
    "npm run develop"
  ],
  "correctAnswer": 1,
  "explanation": "Use 'npm run dev' to start the Vite development server."
}
```

## Quiz Behavior

### Scoring System
- **Pass threshold**: 80% correct answers
- **Immediate feedback**: See results after each question
- **Final score**: Displayed at quiz completion
- **Progress tracking**: Passing quizzes mark chapters as complete

### User Experience
1. **Quiz trigger**: Click "Take Quiz" button on any document page
2. **Question flow**: One question at a time with immediate feedback
3. **Progress indicator**: Shows current question number and total
4. **Results summary**: Final score with option to retake
5. **Progress update**: Passing scores automatically mark chapter complete

## Advanced Configuration

### Custom Pass Threshold
Modify the pass threshold in `src/services/quizService.ts`:

```typescript
// Default is 0.8 (80%)
const PASS_THRESHOLD = 0.7; // Change to 70%
```

### Question Randomization
Add randomization to your quiz loader:

```typescript
// Shuffle questions order
questions: quiz.questions.sort(() => Math.random() - 0.5)
```

### Time Limits
Add timing to questions:

```json
{
  "id": "timed-1",
  "question": "Quick response: What does HTML stand for?",
  "options": ["..."],
  "correctAnswer": 0,
  "timeLimit": 30,  // 30 seconds
  "explanation": "..."
}
```

## Best Practices

### Writing Good Questions

✅ **DO:**
- Use clear, unambiguous language
- Include realistic distractors (wrong answers)
- Test actual understanding, not memorization
- Provide helpful explanations
- Keep questions focused on key concepts

❌ **DON'T:**
- Use trick questions or gotchas
- Make answers too obvious
- Include more than 6 options (cognitive overload)
- Test trivial details
- Write confusing or ambiguous questions

### Question Distribution
- **2-5 questions** per short document
- **5-10 questions** per comprehensive chapter
- **Mix difficulty levels** within each quiz
- **Cover main concepts** not minor details

### Content Alignment
- Questions should **directly relate** to document content
- **Key concepts** should have corresponding quiz questions  
- **Practical application** questions work better than pure theory
- **Progressive difficulty** from basic recall to application

## Managing Quiz Content

### File Organization
```
public/quizzes/
├── introduction.json         # Getting Started section
├── quick-start.json         
├── creating-content.json     # Basic Usage section
├── navigation-setup.json    
└── advanced-config.json     # Advanced section
```

### Version Control
- Include quiz files in your Git repository
- Review quiz changes like any other content
- Consider quiz updates when updating documentation
- Test quiz functionality before deploying

### Bulk Operations
Use scripts for managing multiple quizzes:

```bash
# Find all quizzes missing explanations
grep -L "explanation" public/quizzes/*.json

# Validate JSON syntax for all quizzes  
for file in public/quizzes/*.json; do
  echo "Checking $file..."
  jq . "$file" > /dev/null
done
```

## Troubleshooting

### Quiz Not Loading
1. Check filename matches document ID exactly
2. Validate JSON syntax (use JSONLint or `jq`)
3. Ensure quiz file is in `public/quizzes/`
4. Verify `docId` in quiz matches curriculum

### Questions Not Displaying
1. Check `correctAnswer` is valid index (0-based)
2. Ensure `options` array has 2+ items
3. Verify all required fields are present
4. Check browser console for JavaScript errors

## Next Steps

- Explore [progress tracking](../03_advanced-features/progress-tracking.md) features
- Learn about [notes and bookmarks](../03_advanced-features/notes-bookmarks.md)
- Check out [customization options](../03_advanced-features/customization.md)