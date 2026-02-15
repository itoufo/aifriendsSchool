'use client';

import { useState } from 'react';
import { useNotes } from '../hooks/useNotes';
import './NotesList.css';

export const NotesList = () => {
  const { notes, deleteNote, searchNotes } = useNotes();
  const [searchQuery, setSearchQuery] = useState('');
  
  const displayNotes = searchQuery ? searchNotes(searchQuery) : notes;

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('ja-JP', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (notes.length === 0) {
    return (
      <div className="notes-list-empty">
        <div className="empty-icon">📝</div>
        <h3>ノートはまだありません</h3>
        <p>学習中の気づきやメモを記録して、あとで振り返ることができます</p>
      </div>
    );
  }

  return (
    <div className="notes-list-container">
      <div className="notes-search-bar">
        <input
          type="text"
          placeholder="ノートを検索..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="notes-search-input"
        />
        {searchQuery && (
          <button
            className="search-clear-btn"
            onClick={() => setSearchQuery('')}
            aria-label="クリア"
          >
            ×
          </button>
        )}
      </div>

      {displayNotes.length === 0 ? (
        <div className="no-results">
          <p>「{searchQuery}」に一致するノートが見つかりません</p>
        </div>
      ) : (
        <div className="notes-grid">
          {displayNotes.map((note) => (
            <div key={note.id} className="note-card">
              <div className="note-card-header">
                <span className="note-chapter-id">{note.chapterId}</span>
                <button
                  className="note-delete-btn"
                  onClick={() => deleteNote(note.id)}
                  aria-label="削除"
                >
                  ×
                </button>
              </div>
              
              {note.highlight && (
                <blockquote className="note-highlight-text">
                  {note.highlight.text}
                </blockquote>
              )}
              
              <div className="note-card-content">{note.content}</div>
              
              {note.tags.length > 0 && (
                <div className="note-tags-list">
                  {note.tags.map((tag, index) => (
                    <span key={index} className="note-tag">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
              
              <div className="note-card-footer">
                <span className="note-date">
                  {formatDate(note.createdAt)}
                </span>
                {note.updatedAt !== note.createdAt && (
                  <span className="note-updated">
                    (更新: {formatDate(note.updatedAt)})
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};