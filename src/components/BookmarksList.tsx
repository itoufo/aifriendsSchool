'use client';

import { useRouter } from 'next/navigation';
import { useNotes } from '../hooks/useNotes';
import './BookmarksList.css';

export const BookmarksList = () => {
  const router = useRouter();
  const { bookmarks, deleteBookmark } = useNotes();

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('ja-JP', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const handleNavigate = (chapterId: string) => {
    router.push(`/doc/${chapterId}`);
  };

  if (bookmarks.length === 0) {
    return (
      <div className="bookmarks-list-empty">
        <div className="empty-icon">⭐</div>
        <h3>ブックマークはまだありません</h3>
        <p>学習中の章をブックマークに追加して、後で簡単にアクセスできます</p>
      </div>
    );
  }

  return (
    <div className="bookmarks-list-container">
      <div className="bookmarks-grid">
        {bookmarks.map((bookmark) => (
          <div key={bookmark.id} className="bookmark-card">
            <div className="bookmark-card-header">
              <h3 onClick={() => handleNavigate(bookmark.chapterId)}>
                {bookmark.title}
              </h3>
              <button
                className="bookmark-delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteBookmark(bookmark.id);
                }}
                aria-label="削除"
              >
                ×
              </button>
            </div>
            <div className="bookmark-card-meta">
              <span className="bookmark-date">
                📅 {formatDate(bookmark.createdAt)}
              </span>
            </div>
            {bookmark.note && (
              <p className="bookmark-card-note">{bookmark.note}</p>
            )}
            <button
              className="bookmark-go-btn"
              onClick={() => handleNavigate(bookmark.chapterId)}
            >
              この章へ移動 →
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
