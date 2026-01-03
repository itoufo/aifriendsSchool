import { useState, useEffect, useCallback } from 'react';

export interface Note {
  id: string;
  chapterId: string;
  content: string;
  createdAt: string;
  updatedAt: string;
  tags: string[];
  highlight?: {
    text: string;
    position: number;
  };
}

export interface Bookmark {
  id: string;
  chapterId: string;
  title: string;
  createdAt: string;
  scrollPosition?: number;
  note?: string;
}

const NOTES_KEY = 'ai-school-notes';
const BOOKMARKS_KEY = 'ai-school-bookmarks';

export const useNotes = () => {
  const [notes, setNotes] = useState<Map<string, Note>>(new Map());
  const [bookmarks, setBookmarks] = useState<Map<string, Bookmark>>(new Map());

  useEffect(() => {
    const loadedNotes = localStorage.getItem(NOTES_KEY);
    if (loadedNotes) {
      const parsed = JSON.parse(loadedNotes);
      setNotes(new Map(Object.entries(parsed)));
    }

    const loadedBookmarks = localStorage.getItem(BOOKMARKS_KEY);
    if (loadedBookmarks) {
      const parsed = JSON.parse(loadedBookmarks);
      setBookmarks(new Map(Object.entries(parsed)));
    }
  }, []);

  useEffect(() => {
    if (notes.size > 0) {
      const notesObj = Object.fromEntries(notes);
      localStorage.setItem(NOTES_KEY, JSON.stringify(notesObj));
    }
  }, [notes]);

  useEffect(() => {
    if (bookmarks.size > 0) {
      const bookmarksObj = Object.fromEntries(bookmarks);
      localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(bookmarksObj));
    }
  }, [bookmarks]);

  const createNote = useCallback((chapterId: string, content: string, tags: string[] = [], highlight?: { text: string; position: number }) => {
    const noteId = `note-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const note: Note = {
      id: noteId,
      chapterId,
      content,
      tags,
      highlight,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    setNotes(prev => {
      const newNotes = new Map(prev);
      newNotes.set(noteId, note);
      return newNotes;
    });

    return noteId;
  }, []);

  const updateNote = useCallback((noteId: string, content: string, tags?: string[]) => {
    setNotes(prev => {
      const newNotes = new Map(prev);
      const existing = newNotes.get(noteId);
      
      if (existing) {
        newNotes.set(noteId, {
          ...existing,
          content,
          tags: tags || existing.tags,
          updatedAt: new Date().toISOString(),
        });
      }
      
      return newNotes;
    });
  }, []);

  const deleteNote = useCallback((noteId: string) => {
    setNotes(prev => {
      const newNotes = new Map(prev);
      newNotes.delete(noteId);
      return newNotes;
    });
  }, []);

  const getNotesByChapter = useCallback((chapterId: string): Note[] => {
    return Array.from(notes.values()).filter(note => note.chapterId === chapterId);
  }, [notes]);

  const searchNotes = useCallback((query: string): Note[] => {
    const lowercaseQuery = query.toLowerCase();
    return Array.from(notes.values()).filter(note => 
      note.content.toLowerCase().includes(lowercaseQuery) ||
      note.tags.some(tag => tag.toLowerCase().includes(lowercaseQuery)) ||
      note.highlight?.text.toLowerCase().includes(lowercaseQuery)
    );
  }, [notes]);

  const createBookmark = useCallback((chapterId: string, title: string, scrollPosition?: number, note?: string) => {
    const bookmarkId = `bookmark-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const bookmark: Bookmark = {
      id: bookmarkId,
      chapterId,
      title,
      scrollPosition,
      note,
      createdAt: new Date().toISOString(),
    };

    setBookmarks(prev => {
      const newBookmarks = new Map(prev);
      newBookmarks.set(bookmarkId, bookmark);
      return newBookmarks;
    });

    return bookmarkId;
  }, []);

  const deleteBookmark = useCallback((bookmarkId: string) => {
    setBookmarks(prev => {
      const newBookmarks = new Map(prev);
      newBookmarks.delete(bookmarkId);
      return newBookmarks;
    });
  }, []);

  const getBookmarksByChapter = useCallback((chapterId: string): Bookmark[] => {
    return Array.from(bookmarks.values()).filter(bookmark => bookmark.chapterId === chapterId);
  }, [bookmarks]);

  const isBookmarked = useCallback((chapterId: string): boolean => {
    return Array.from(bookmarks.values()).some(bookmark => bookmark.chapterId === chapterId);
  }, [bookmarks]);

  const exportNotes = useCallback((format: 'json' | 'markdown' = 'json') => {
    if (format === 'json') {
      const data = {
        notes: Object.fromEntries(notes),
        bookmarks: Object.fromEntries(bookmarks),
        exportedAt: new Date().toISOString(),
      };
      return JSON.stringify(data, null, 2);
    } else {
      let markdown = '# AI School Notes Export\n\n';
      markdown += `*Exported on ${new Date().toLocaleDateString()}*\n\n`;
      
      const notesByChapter = new Map<string, Note[]>();
      notes.forEach(note => {
        if (!notesByChapter.has(note.chapterId)) {
          notesByChapter.set(note.chapterId, []);
        }
        notesByChapter.get(note.chapterId)?.push(note);
      });
      
      notesByChapter.forEach((chapterNotes, chapterId) => {
        markdown += `## Chapter: ${chapterId}\n\n`;
        chapterNotes.forEach(note => {
          markdown += `### Note (${new Date(note.createdAt).toLocaleDateString()})\n`;
          if (note.highlight) {
            markdown += `> **Highlight:** ${note.highlight.text}\n\n`;
          }
          markdown += `${note.content}\n`;
          if (note.tags.length > 0) {
            markdown += `\n**Tags:** ${note.tags.join(', ')}\n`;
          }
          markdown += '\n---\n\n';
        });
      });
      
      if (bookmarks.size > 0) {
        markdown += '## Bookmarks\n\n';
        bookmarks.forEach(bookmark => {
          markdown += `- **${bookmark.title}** (Chapter: ${bookmark.chapterId})\n`;
          if (bookmark.note) {
            markdown += `  ${bookmark.note}\n`;
          }
        });
      }
      
      return markdown;
    }
  }, [notes, bookmarks]);

  const importNotes = useCallback((jsonData: string) => {
    try {
      const data = JSON.parse(jsonData);
      if (data.notes) {
        setNotes(new Map(Object.entries(data.notes)));
      }
      if (data.bookmarks) {
        setBookmarks(new Map(Object.entries(data.bookmarks)));
      }
    } catch (error) {
      console.error('Failed to import notes:', error);
      throw new Error('Invalid notes data format');
    }
  }, []);

  const clearAllNotes = useCallback(() => {
    setNotes(new Map());
    setBookmarks(new Map());
    localStorage.removeItem(NOTES_KEY);
    localStorage.removeItem(BOOKMARKS_KEY);
  }, []);

  return {
    notes: Array.from(notes.values()),
    bookmarks: Array.from(bookmarks.values()),
    createNote,
    updateNote,
    deleteNote,
    getNotesByChapter,
    searchNotes,
    createBookmark,
    deleteBookmark,
    getBookmarksByChapter,
    isBookmarked,
    exportNotes,
    importNotes,
    clearAllNotes,
  };
};