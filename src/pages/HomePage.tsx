import { Link } from 'react-router-dom';
import { curriculum } from '../data/curriculum';
import { BookmarksList } from '../components/BookmarksList';
import { NotesList } from '../components/NotesList';
import { useProgress } from '../hooks/useProgress';
import { ProgressBar } from '../components/ProgressBar';
import './HomePage.css';

// レベルごとのサムネイル画像マッピング
const levelThumbnails: Record<number, string> = {
  0: '/images/illustrations/level0-beginner-start.jpg',
  1: '/images/illustrations/level1-ai-literacy.jpg',
  2: '/images/illustrations/level2-advanced-prompt.jpg',
  3: '/images/illustrations/level3-custom-ai.jpg',
  4: '/images/illustrations/level4-infrastructure.jpg',
  5: '/images/illustrations/level5-corporate-strategy.jpg',
};

export const HomePage = () => {
  const { stats } = useProgress();
  const totalChapters = curriculum.reduce((acc, section) => acc + section.items.length, 0);

  return (
    <div className="home-page">
      {/* ヒーローセクション */}
      <section className="hero">
        <div className="hero-content">
          <img
            src="/images/logo.png"
            alt="AI Friends School Logo"
            className="hero-logo"
          />
          <h1>AI Friends School</h1>
          <h2 className="hero-subtitle">包括的AIスクールカリキュラム</h2>
          <p className="hero-description">
            初心者から経営者まで、5つのレベル別に体系的にAI活用能力を育成。<br/>
            真に「稼げる」人材、そして未来を創造できる人材になるための実践的教育プログラム
          </p>
          <div className="hero-badges">
            <span className="badge">🎯 5レベル制</span>
            <span className="badge">💼 実践重視</span>
            <span className="badge">👥 個人〜経営者</span>
            <span className="badge">📚 完全体系化</span>
          </div>
          <div className="hero-cta">
            <Link to="/doc/ai-literacy-and-ethics" className="cta-btn primary">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5,3 19,12 5,21" />
              </svg>
              今すぐ始める
            </Link>
            <Link to="/doc/getting-started" className="cta-btn secondary">
              カリキュラムを見る
            </Link>
          </div>
        </div>
        <div className="scroll-indicator">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="7,10 12,15 17,10" />
          </svg>
        </div>
      </section>

      {/* レベル概要セクション */}
      <section className="level-overview">
        <h2>5つのレベル構成</h2>
        <p className="section-subtitle">あなたの現在地から、AIマスターへの道を歩もう</p>
        <div className="levels-grid">
          {curriculum.map((section, index) => (
            <div key={section.id} className={`level-card level-${index + 1}`}>
              <div className="level-thumbnail-wrapper">
                <img
                  src={levelThumbnails[index + 1] || levelThumbnails[1]}
                  alt={section.title}
                  className="level-thumbnail"
                  loading="lazy"
                />
              </div>
              <div className="level-header">
                <span className="level-number">Lv.{index + 1}</span>
                <h3>{section.title.split('：')[1] || section.title}</h3>
              </div>
              <div className="level-info">
                <span className="level-target">{section.targetAudience}</span>
                <p className="level-duration">📅 {section.duration}</p>
                <p className="level-description">{section.description}</p>
              </div>
              <ul className="level-items">
                {section.items.slice(0, 2).map((item) => (
                  <li key={item.id}>
                    <Link to={`/doc/${item.id}`}>{item.title}</Link>
                  </li>
                ))}
                {section.items.length > 2 && (
                  <li className="more-items">
                    +{section.items.length - 2}個のトピック
                  </li>
                )}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* 特徴セクション */}
      <section className="features">
        <h2>AI Friends Schoolの特徴</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
              </svg>
            </div>
            <h3>段階別学習</h3>
            <p>初心者から経営者まで、現在のレベルに応じた最適な学習パスで効率的にスキルアップ</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
            </div>
            <h3>実践重視</h3>
            <p>業務で即座に活用できる具体的なスキルと技術を習得。理論だけでなく実践力を重視</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>
            <h3>ヒューマンスキル統合</h3>
            <p>AIツールと人間固有の能力を融合した新時代の働き方をマスター</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
              </svg>
            </div>
            <h3>価値創造重視</h3>
            <p>単なる効率化を超えた、競争優位性と新規事業の創造。真に「稼げる」人材へ</p>
          </div>
        </div>
      </section>

      {/* 学習進捗セクション */}
      <section className="progress-overview">
        <h2>📊 学習進捗</h2>
        <ProgressBar
          current={stats.completedChapters}
          total={totalChapters}
          label="全体の進捗"
          color="success"
          size="large"
        />
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-value">{stats.completedChapters}</span>
            <span className="stat-label">完了した章</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{Math.floor(stats.totalTimeSpent / 60)}</span>
            <span className="stat-label">学習時間（分）</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.currentStreak}</span>
            <span className="stat-label">連続学習日数</span>
          </div>
        </div>
      </section>

      {/* ブックマークセクション */}
      <section className="bookmarks-section">
        <h2>⭐ ブックマーク</h2>
        <BookmarksList />
      </section>

      {/* ノートセクション */}
      <section className="notes-section">
        <h2>📝 ノート</h2>
        <NotesList />
      </section>

      {/* クイックスタートセクション */}
      <section className="quick-start">
        <h2>🚀 学習を始める</h2>
        <div className="start-options">
          <div className="start-option">
            <h3>🎯 レベル診断から始める</h3>
            <p>あなたの現在のAIリテラシーレベルを診断し、最適な学習パスを提案します</p>
            <Link to="/doc/ai-literacy-and-ethics" className="start-btn primary">
              レベル1から始める
            </Link>
          </div>
          <div className="start-option">
            <h3>📚 カリキュラム全体を確認</h3>
            <p>5つのレベル構成と学習ロードマップを詳しく確認してから始めましょう</p>
            <div className="start-buttons">
              <Link to="/doc/ai-literacy-and-ethics" className="start-btn secondary">
                Lv.1 初心者
              </Link>
              <Link to="/doc/advanced-prompt-engineering" className="start-btn secondary">
                Lv.2 中級者
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
