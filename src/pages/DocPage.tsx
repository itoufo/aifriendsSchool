import { useState, useEffect } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { MarkdownViewer } from '../components/MarkdownViewer';
import { QuizModal } from '../components/QuizModal';
import { NotesPanel } from '../components/NotesPanel';
import { getDocById, getNextDoc } from '../data/curriculum';
import { QuizService } from '../services/quizService';
import { useProgress } from '../hooks/useProgress';
import type { Quiz } from '../data/quiz.types';
import './DocPage.css';

export const DocPage = () => {
  const { docId } = useParams<{ docId: string }>();
  const [isQuizOpen, setIsQuizOpen] = useState(false);
  const [isNotesOpen, setIsNotesOpen] = useState(false);
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [loadingQuiz, setLoadingQuiz] = useState(true);
  const { markAsVisited, updateTimeSpent, getChapterProgress, markAsCompleted } = useProgress();

  useEffect(() => {
    if (docId) {
      markAsVisited(docId);
      // クイズを動的に読み込む
      setLoadingQuiz(true);
      QuizService.loadQuiz(docId)
        .then(quizData => {
          setQuiz(quizData);
        })
        .finally(() => {
          setLoadingQuiz(false);
        });
    }
  }, [docId, markAsVisited]);

  useEffect(() => {
    if (docId) {
      return () => {
        updateTimeSpent(docId);
      };
    }
  }, [docId, updateTimeSpent]);

  if (!docId) {
    return <Navigate to="/" replace />;
  }

  const doc = getDocById(docId);
  const progress = getChapterProgress(docId);

  if (!doc) {
    return (
      <div className="not-found">
        <h2>ページが見つかりません</h2>
        <p>指定されたドキュメントは存在しません。</p>
      </div>
    );
  }

  const nextDoc = getNextDoc(docId);

  const handleQuizComplete = (passed: boolean) => {
    if (passed && docId) {
      markAsCompleted(docId);
    }
  };

  return (
    <>
      <div className="doc-header-controls">
        <button
          className="notes-toggle-button"
          onClick={() => setIsNotesOpen(!isNotesOpen)}
          aria-label="ノートを開く"
        >
          📝 ノート
        </button>
        {progress?.completed && (
          <span className="completion-badge">✅ 完了済み</span>
        )}
      </div>

      <MarkdownViewer filePath={doc.path} title={doc.title} />

      {!loadingQuiz && quiz && (
        <div className="quiz-section">
          <div className="quiz-section-content">
            <div className="quiz-section-info">
              <h3>確認クイズ</h3>
              <p>
                この章の理解度を確認しましょう。
                <span className="quiz-count">{quiz.questions.length}問</span>
                <span className="quiz-pass-info">（8割以上で合格）</span>
              </p>
            </div>
            <button
              className="quiz-start-button"
              onClick={() => setIsQuizOpen(true)}
            >
              クイズを開始
            </button>
          </div>
        </div>
      )}

      {quiz && (
        <QuizModal
          quiz={quiz}
          nextDoc={nextDoc}
          isOpen={isQuizOpen}
          onClose={() => setIsQuizOpen(false)}
          onComplete={handleQuizComplete}
        />
      )}

      {docId && (
        <NotesPanel
          chapterId={docId}
          isOpen={isNotesOpen}
          onClose={() => setIsNotesOpen(false)}
        />
      )}
    </>
  );
};
