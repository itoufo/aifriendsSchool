import { Link } from 'react-router-dom';
import { curriculum } from '../data/curriculum';
import { BookmarksList } from '../components/BookmarksList';
import { NotesList } from '../components/NotesList';
import { useProgress } from '../hooks/useProgress';
import { ProgressBar } from '../components/ProgressBar';
import './HomePage.css';

export const HomePage = () => {
  const { stats } = useProgress();
  const totalChapters = curriculum.reduce((acc, section) => acc + section.items.length, 0);
  return (
    <div className="home-page">
      <section className="hero">
        <h1>AI Friends School</h1>
        <h2 className="hero-subtitle">包括的AIスクールカリキュラム</h2>
        <p className="hero-description">
          初心者から経営者まで、5つのレベル別に体系的にAI活用能力を育成。<br/>
          真に「稼げる」人材、そして未来を創造できる人材になるための実践的教育プログラム
        </p>
        <div className="hero-badges">
          <span className="badge">5レベル制</span>
          <span className="badge">実践重視</span>
          <span className="badge">個人〜経営者</span>
          <span className="badge">完全体系化</span>
        </div>
      </section>

      <section className="level-overview">
        <h2>5つのレベル構成</h2>
        <div className="levels-grid">
          {curriculum.map((section, index) => (
            <div key={section.id} className={`level-card level-${index + 1}`}>
              <div className="level-header">
                <span className="level-number">Level {index + 1}</span>
                <h3>{section.title.split('：')[1] || section.title}</h3>
              </div>
              <div className="level-info">
                <p className="level-target">{section.targetAudience}</p>
                <p className="level-duration">{section.duration}</p>
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

      <section className="features">
        <h2>AI Friends Schoolの特徴</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 11H5a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h4" />
                <path d="M15 11h4a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-4" />
                <path d="M9 7V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v4" />
              </svg>
            </div>
            <h3>段階別学習</h3>
            <p>初心者から経営者まで、現在のレベルに応じた最適な学習パス</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
              </svg>
            </div>
            <h3>実践重視</h3>
            <p>業務で即座に活用できる具体的なスキルと技術を習得</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>
            <h3>ヒューマンスキル統合</h3>
            <p>AIツールと人間固有の能力を融合した新時代の働き方</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
              </svg>
            </div>
            <h3>価値創造重視</h3>
            <p>単なる効率化を超えた、競争優位性と新規事業の創造</p>
          </div>
        </div>
      </section>

      <section className="progress-overview">
        <h2>学習進捗</h2>
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
            <span className="stat-label">完了章</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{Math.floor(stats.totalTimeSpent / 60)}</span>
            <span className="stat-label">学習時間(分)</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.currentStreak}</span>
            <span className="stat-label">連続学習日数</span>
          </div>
        </div>
      </section>

      <section className="bookmarks-section">
        <h2>⭐ ブックマーク</h2>
        <BookmarksList />
      </section>

      <section className="notes-section">
        <h2>📝 ノート</h2>
        <NotesList />
      </section>

      <section className="quick-start">
        <h2>学習を始める</h2>
        <div className="start-options">
          <div className="start-option">
            <h3>🎯 レベル診断から始める</h3>
            <p>あなたの現在のAIリテラシーレベルを診断し、最適な学習パスを提案</p>
            <Link to="/doc/ai-literacy-and-ethics" className="start-btn primary">
              レベル1から始める
            </Link>
          </div>
          <div className="start-option">
            <h3>📚 カリキュラム全体を確認</h3>
            <p>5つのレベル構成と学習ロードマップを詳しく確認</p>
            <div className="start-buttons">
              <Link to="/doc/ai-literacy-and-ethics" className="start-btn secondary">
                レベル1: 初心者
              </Link>
              <Link to="/doc/advanced-prompt-engineering" className="start-btn secondary">
                レベル2: 中級者
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
