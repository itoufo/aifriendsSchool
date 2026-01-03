# Notes & Bookmarks

Enable users to take notes, highlight content, and bookmark important sections.

## Overview

The notes and bookmarks system provides:
- 📝 **Personal notes** linked to specific chapters
- 🔖 **Quick bookmarks** for easy navigation
- 🎨 **Text highlighting** with persistent storage
- 🏷️ **Tagging system** for organization
- 💾 **Local storage** - all data stays private

## Note-Taking Features

### Basic Note Structure
```typescript
interface Note {
  id: string;              // Unique note identifier
  chapterId: string;       // Associated document ID
  content: string;         // Note text content
  createdAt: string;       // Creation timestamp
  updatedAt: string;       // Last modification timestamp
  tags: string[];          // Optional tags for organization
  highlight?: {            // Optional highlighted text
    text: string;          // The highlighted text
    position: number;      // Position in document
  };
}
```

### Creating Notes
Users can create notes in several ways:

1. **Chapter Notes**: General notes for the entire chapter
2. **Highlight Notes**: Notes attached to specific text selections
3. **Quick Notes**: Fast note-taking during reading

## Using the Notes Hook

### Basic Usage
```typescript
import { useNotes } from '../hooks/useNotes';

function NotesComponent() {
  const {
    notes,              // All notes
    addNote,           // Add new note
    updateNote,        // Update existing note
    deleteNote,        // Delete note
    getNotesForChapter // Get notes for specific chapter
  } = useNotes();

  // Get notes for current chapter
  const chapterNotes = getNotesForChapter('introduction');

  // Add a new note
  const handleAddNote = () => {
    addNote({
      chapterId: 'introduction',
      content: 'This is an important concept to remember',
      tags: ['important', 'concept']
    });
  };
}
```

### Note Management
```typescript
// Update an existing note
updateNote('note-id', {
  content: 'Updated note content',
  tags: ['updated', 'important'],
  updatedAt: new Date().toISOString()
});

// Delete a note
deleteNote('note-id');

// Get notes with specific tags
const importantNotes = notes.filter(note => 
  note.tags.includes('important')
);
```

## Bookmark System

### Bookmark Structure
```typescript
interface Bookmark {
  id: string;           // Unique identifier
  chapterId: string;    // Document ID
  title: string;        // Bookmark title
  createdAt: string;    // Creation timestamp
  section?: string;     // Optional section within document
}
```

### Managing Bookmarks
```typescript
import { useNotes } from '../hooks/useNotes';

function BookmarkComponent() {
  const { 
    bookmarks,
    addBookmark, 
    deleteBookmark,
    getBookmarksForChapter 
  } = useNotes();

  // Add a bookmark
  const handleBookmark = () => {
    addBookmark({
      chapterId: 'quiz-system',
      title: 'Quiz Configuration Examples',
      section: 'Basic Multiple Choice'
    });
  };
}
```

## Text Highlighting

### Highlight Implementation
```typescript
// Add highlighted text with note
const addHighlightNote = (selectedText: string, position: number) => {
  addNote({
    chapterId: getCurrentChapterId(),
    content: 'My thoughts on this highlighted text...',
    tags: ['highlight'],
    highlight: {
      text: selectedText,
      position: position
    }
  });
};

// Get all highlights for current chapter
const getHighlights = (chapterId: string) => {
  return notes
    .filter(note => note.chapterId === chapterId && note.highlight)
    .map(note => note.highlight);
};
```

### Persistent Highlighting
```css
/* CSS for highlighted text */
.highlighted-text {
  background-color: #fff3cd;
  border-bottom: 2px solid #ffc107;
  cursor: pointer;
  position: relative;
}

.highlighted-text:hover::after {
  content: "📝 View note";
  position: absolute;
  top: -30px;
  left: 0;
  background: #333;
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
}
```

## UI Components

### Notes Panel
```tsx
function NotesPanel({ chapterId }: { chapterId: string }) {
  const { getNotesForChapter, addNote, updateNote, deleteNote } = useNotes();
  const [isOpen, setIsOpen] = useState(false);
  const [newNote, setNewNote] = useState('');
  
  const chapterNotes = getNotesForChapter(chapterId);

  return (
    <div className="notes-panel">
      <button onClick={() => setIsOpen(!isOpen)}>
        📝 Notes ({chapterNotes.length})
      </button>
      
      {isOpen && (
        <div className="notes-content">
          <textarea
            value={newNote}
            onChange={(e) => setNewNote(e.target.value)}
            placeholder="Add a note for this chapter..."
          />
          <button onClick={() => {
            if (newNote.trim()) {
              addNote({ chapterId, content: newNote.trim() });
              setNewNote('');
            }
          }}>
            Add Note
          </button>
          
          <div className="existing-notes">
            {chapterNotes.map(note => (
              <NoteItem 
                key={note.id} 
                note={note}
                onUpdate={updateNote}
                onDelete={deleteNote}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

### Bookmark Sidebar
```tsx
function BookmarkSidebar() {
  const { bookmarks, deleteBookmark } = useNotes();
  const navigate = useNavigate();

  return (
    <div className="bookmark-sidebar">
      <h3>🔖 Bookmarks</h3>
      {bookmarks.length === 0 ? (
        <p>No bookmarks yet</p>
      ) : (
        <ul className="bookmark-list">
          {bookmarks.map(bookmark => (
            <li key={bookmark.id} className="bookmark-item">
              <button
                onClick={() => navigate(`/doc/${bookmark.chapterId}`)}
                className="bookmark-link"
              >
                {bookmark.title}
              </button>
              <button
                onClick={() => deleteBookmark(bookmark.id)}
                className="bookmark-delete"
                aria-label="Delete bookmark"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

## Advanced Features

### Search Notes
```typescript
const searchNotes = (query: string) => {
  const lowercaseQuery = query.toLowerCase();
  return notes.filter(note =>
    note.content.toLowerCase().includes(lowercaseQuery) ||
    note.tags.some(tag => tag.toLowerCase().includes(lowercaseQuery))
  );
};

// Usage in component
const [searchQuery, setSearchQuery] = useState('');
const filteredNotes = useMemo(() => 
  searchQuery ? searchNotes(searchQuery) : notes
, [searchQuery, notes]);
```

### Note Export
```typescript
const exportNotes = () => {
  const exportData = {
    exportDate: new Date().toISOString(),
    notes: notes,
    bookmarks: bookmarks
  };
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], {
    type: 'application/json'
  });
  
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `knowledge-notes-${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  URL.revokeObjectURL(url);
};
```

### Note Synchronization
```typescript
// Future feature: sync with external services
interface NoteSyncService {
  uploadNotes(notes: Note[]): Promise<void>;
  downloadNotes(): Promise<Note[]>;
  syncConflicts(local: Note[], remote: Note[]): Promise<Note[]>;
}

const syncNotes = async (service: NoteSyncService) => {
  try {
    const remoteNotes = await service.downloadNotes();
    const conflicts = await service.syncConflicts(notes, remoteNotes);
    // Handle merge conflicts...
  } catch (error) {
    console.error('Sync failed:', error);
  }
};
```

## Data Storage Schema

### LocalStorage Structure
```
localStorage['knowledgeFramework_notes'] = {
  "note-1": {
    "id": "note-1",
    "chapterId": "introduction", 
    "content": "Key concepts to remember",
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-01-15T10:30:00Z",
    "tags": ["important"]
  },
  // ... more notes
}

localStorage['knowledgeFramework_bookmarks'] = {
  "bookmark-1": {
    "id": "bookmark-1",
    "chapterId": "quiz-system",
    "title": "Quiz Examples",
    "createdAt": "2024-01-15T11:00:00Z"
  },
  // ... more bookmarks
}
```

## Best Practices

### Note Organization
- **Use consistent tagging** for easy categorization
- **Keep notes concise** and focused
- **Regular cleanup** of outdated notes
- **Export backups** periodically

### Performance Considerations
- **Limit note size** (recommend 1000 characters max)
- **Paginate notes** for chapters with many notes
- **Lazy load** note content when needed
- **Index by chapter** for fast retrieval

### User Experience
- **Auto-save** notes as user types (with debouncing)
- **Keyboard shortcuts** for quick note-taking
- **Visual indicators** for chapters with notes
- **Undo functionality** for accidental deletions

## Troubleshooting

### Notes Not Saving
1. Check localStorage quota and permissions
2. Verify JSON serialization isn't failing
3. Test in different browsers
4. Check for JavaScript errors

### Performance Issues
1. Monitor localStorage size usage
2. Implement note pagination for large datasets
3. Consider note archiving for old content
4. Optimize note search algorithms

### Data Loss Prevention
1. Implement auto-backup to file
2. Add confirmation dialogs for deletions
3. Provide note export functionality
4. Consider cloud sync options

## Next Steps

- Learn about [customization options](customization.md) to style notes and bookmarks
- Explore [deployment considerations](../04_deployment/building.md) for production use
- Check out the [hosting guide](../04_deployment/hosting.md) for different platforms