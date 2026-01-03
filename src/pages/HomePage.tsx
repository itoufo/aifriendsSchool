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
        <h1>AI活用人材育成カリキュラム</h1>
        <p className="hero-subtitle">
          健全AI教育協会（HAIIA）が提唱する次世代AI教育プログラム
        </p>
        <div className="hero-badges">
          <span className="badge">3段階学習</span>
          <span className="badge">4つの力</span>
          <span className="badge">Lv1-Lv5評価</span>
        </div>
      </section>

      <section className="four-powers">
        <h2>AI時代に必要な4つの力</h2>
        <div className="powers-grid">
          <div className="power-card">
            <div className="power-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <h3>コミュニケーション力</h3>
            <p>人とAIの対話を設計・調整し、情報を構造化して合意を導く力</p>
          </div>
          <div className="power-card">
            <div className="power-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 20V10M18 20V4M6 20v-4" />
              </svg>
            </div>
            <h3>言語力</h3>
            <p>目的に応じた文章を明確に書き、AIとの対話を論理的に構築する力</p>
          </div>
          <div className="power-card">
            <div className="power-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 6v6l4 2" />
              </svg>
            </div>
            <h3>セルフコーチング力</h3>
            <p>自分で目標を設定し、学習プロセスを振り返り改善する力</p>
          </div>
          <div className="power-card">
            <div className="power-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                <line x1="16" y1="2" x2="16" y2="6" />
                <line x1="8" y1="2" x2="8" y2="6" />
                <line x1="3" y1="10" x2="21" y2="10" />
              </svg>
            </div>
            <h3>PM力</h3>
            <p>タスクを分解し、リソースを配分して目標を達成する力</p>
          </div>
        </div>
      </section>

      <section className="curriculum-overview">
        <h2>カリキュラム構成</h2>
        <div className="stages-grid">
          {curriculum.slice(1, 4).map((section, index) => (
            <div key={section.id} className={`stage-card stage-${index + 1}`}>
              <div className="stage-header">
                <span className="stage-number">{index + 1}</span>
                <h3>{section.title.replace(/第\d段階：/, '')}</h3>
              </div>
              <ul className="stage-items">
                {section.items.map((item) => (
                  <li key={item.id}>
                    <Link to={`/doc/${item.id}`}>{item.title}</Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
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
        <div className="start-buttons">
          <Link to="/doc/curriculum-overview" className="start-btn primary">
            カリキュラム概要を見る
          </Link>
          <Link to="/doc/ai-literacy" className="start-btn secondary">
            入門コースから始める
          </Link>
        </div>
      </section>
    </div>
  );
};
