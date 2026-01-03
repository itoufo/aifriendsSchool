import { useState, useEffect } from 'react';
import { useNotes } from '../hooks/useNotes';
import './NotesPanel.css';

interface NotesPanelProps {
  chapterId: string;
  isOpen: boolean;
  onClose: () => void;
}

export const NotesPanel = ({ chapterId, isOpen, onClose }: NotesPanelProps) => {
  const {
    notes,
    createNote,
    updateNote,
    deleteNote,
    getNotesByChapter,
    createBookmark,
    deleteBookmark,
    isBookmarked,
    getBookmarksByChapter,
  } = useNotes();

  const [noteContent, setNoteContent] = useState('');
  const [editingNoteId, setEditingNoteId] = useState<string | null>(null);
  const [tags, setTags] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'notes' | 'bookmarks'>('notes');

  const chapterNotes = getNotesByChapter(chapterId);
  const chapterBookmarks = getBookmarksByChapter(chapterId);
  const hasBookmark = isBookmarked(chapterId);

  useEffect(() => {
    if (!isOpen) {
      setNoteContent('');
      setEditingNoteId(null);
      setTags('');
    }
  }, [isOpen]);

  const handleSaveNote = () => {
    if (!noteContent.trim()) return;

    const tagArray = tags.split(',').map(t => t.trim()).filter(Boolean);

    if (editingNoteId) {
      updateNote(editingNoteId, noteContent, tagArray);
      setEditingNoteId(null);
    } else {
      createNote(chapterId, noteContent, tagArray);
    }

    setNoteContent('');
    setTags('');
  };

  const handleEditNote = (noteId: string) => {
    const note = notes.find(n => n.id === noteId);
    if (note) {
      setNoteContent(note.content);
      setTags(note.tags.join(', '));
      setEditingNoteId(noteId);
    }
  };

  const handleToggleBookmark = () => {
    if (hasBookmark) {
      const bookmark = chapterBookmarks[0];
      if (bookmark) {
        deleteBookmark(bookmark.id);
      }
    } else {
      const title = document.querySelector('.doc-title-bar')?.textContent || 'Untitled';
      createBookmark(chapterId, title);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('ja-JP', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (!isOpen) return null;

  return (
    <div className="notes-panel">
      <div className="notes-panel-header">
        <div className="notes-panel-tabs">
          <button
            className={`tab-button ${activeTab === 'notes' ? 'active' : ''}`}
            onClick={() => setActiveTab('notes')}
          >
            ノート ({chapterNotes.length})
          </button>
          <button
            className={`tab-button ${activeTab === 'bookmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('bookmarks')}
          >
            ブックマーク
          </button>
        </div>
        <button className="close-button" onClick={onClose} aria-label="閉じる">
          ×
        </button>
      </div>

      <div className="notes-panel-content">
        {activeTab === 'notes' ? (
          <>
            <div className="note-input-section">
              <textarea
                className="note-input"
                placeholder="ノートを入力..."
                value={noteContent}
                onChange={(e) => setNoteContent(e.target.value)}
                rows={4}
              />
              <input
                className="tags-input"
                type="text"
                placeholder="タグ (カンマ区切り)"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
              />
              <div className="note-actions">
                {editingNoteId && (
                  <button
                    className="cancel-button"
                    onClick={() => {
                      setEditingNoteId(null);
                      setNoteContent('');
                      setTags('');
                    }}
                  >
                    キャンセル
                  </button>
                )}
                <button
                  className="save-button"
                  onClick={handleSaveNote}
                  disabled={!noteContent.trim()}
                >
                  {editingNoteId ? '更新' : '保存'}
                </button>
              </div>
            </div>

            <div className="notes-list">
              {chapterNotes.length === 0 ? (
                <div className="empty-state">
                  <p>まだノートがありません</p>
                  <p className="hint">学習中の気づきやメモを記録しましょう</p>
                </div>
              ) : (
                chapterNotes.map((note) => (
                  <div key={note.id} className="note-item">
                    <div className="note-header">
                      <span className="note-date">{formatDate(note.createdAt)}</span>
                      <div className="note-item-actions">
                        <button
                          className="edit-button"
                          onClick={() => handleEditNote(note.id)}
                          aria-label="編集"
                        >
                          ✏️
                        </button>
                        <button
                          className="delete-button"
                          onClick={() => deleteNote(note.id)}
                          aria-label="削除"
                        >
                          🗑️
                        </button>
                      </div>
                    </div>
                    {note.highlight && (
                      <blockquote className="note-highlight">
                        {note.highlight.text}
                      </blockquote>
                    )}
                    <div className="note-content">{note.content}</div>
                    {note.tags.length > 0 && (
                      <div className="note-tags">
                        {note.tags.map((tag, index) => (
                          <span key={index} className="tag">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </>
        ) : (
          <div className="bookmarks-section">
            <button
              className={`bookmark-toggle ${hasBookmark ? 'bookmarked' : ''}`}
              onClick={handleToggleBookmark}
            >
              {hasBookmark ? '⭐ ブックマーク済み' : '☆ ブックマークに追加'}
            </button>

            <div className="bookmarks-list">
              {chapterBookmarks.length === 0 ? (
                <div className="empty-state">
                  <p>このチャプターはブックマークされていません</p>
                </div>
              ) : (
                chapterBookmarks.map((bookmark) => (
                  <div key={bookmark.id} className="bookmark-item">
                    <div className="bookmark-info">
                      <h4>{bookmark.title}</h4>
                      <span className="bookmark-date">
                        {formatDate(bookmark.createdAt)}
                      </span>
                    </div>
                    {bookmark.note && (
                      <p className="bookmark-note">{bookmark.note}</p>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};