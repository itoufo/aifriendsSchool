import { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { curriculum } from '../data/curriculum';
import type { Section } from '../data/curriculum';
import { useProgress } from '../hooks/useProgress';
import { useNotes } from '../hooks/useNotes';
import { ProgressBar } from './ProgressBar';
import './Sidebar.css';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const SectionItem = ({ section }: { section: Section }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="sidebar-section">
      <button
        className="section-header"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span className="section-title">{section.title}</span>
        <span className={`chevron ${isExpanded ? 'expanded' : ''}`}>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
            <path d="M4 2L8 6L4 10" strokeWidth="2" stroke="currentColor" fill="none" />
          </svg>
        </span>
      </button>
      {isExpanded && (
        <ul className="section-items">
          {section.items.map((item) => (
            <li key={item.id}>
              <NavLink
                to={`/doc/${item.id}`}
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'active' : ''}`
                }
              >
                {item.title}
              </NavLink>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export const Sidebar = ({ isOpen, onClose }: SidebarProps) => {
  const { stats } = useProgress();
  const { bookmarks } = useNotes();
  const navigate = useNavigate();
  const [showBookmarks, setShowBookmarks] = useState(false);
  return (
    <>
      <div
        className={`sidebar-overlay ${isOpen ? 'visible' : ''}`}
        onClick={onClose}
      />
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>AIフレンズ</h2>
          <button className="close-btn" onClick={onClose} aria-label="Close sidebar">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="sidebar-progress">
          <ProgressBar
            current={stats.completedChapters}
            total={stats.totalChapters || curriculum.reduce((acc, section) => acc + section.items.length, 0)}
            label="学習進捗"
            color="success"
            size="small"
          />
        </div>
        
        <div className="sidebar-quick-links">
          <NavLink
            to="/"
            className={({ isActive }) => `home-link ${isActive ? 'active' : ''}`}
            onClick={onClose}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
              <polyline points="9 22 9 12 15 12 15 22"/>
            </svg>
            <span>ホーム</span>
          </NavLink>
          
          <button
            className={`bookmarks-toggle ${showBookmarks ? 'expanded' : ''}`}
            onClick={() => setShowBookmarks(!showBookmarks)}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
            </svg>
            <span>ブックマーク ({bookmarks.length})</span>
            <span className={`chevron ${showBookmarks ? 'expanded' : ''}`}>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                <path d="M4 2L8 6L4 10" strokeWidth="2" stroke="currentColor" fill="none" />
              </svg>
            </span>
          </button>
          
          {showBookmarks && (
            <div className="bookmarks-list-sidebar">
              {bookmarks.length === 0 ? (
                <div className="no-bookmarks">ブックマークはありません</div>
              ) : (
                bookmarks.map((bookmark) => (
                  <div
                    key={bookmark.id}
                    className="bookmark-item-sidebar"
                    onClick={() => {
                      navigate(`/doc/${bookmark.chapterId}`);
                      onClose();
                    }}
                  >
                    <span className="bookmark-title">{bookmark.title}</span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
        
        <nav className="sidebar-nav">
          {curriculum.map((section) => (
            <SectionItem key={section.id} section={section} />
          ))}
        </nav>
      </aside>
    </>
  );
};
