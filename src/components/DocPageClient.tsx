'use client';

import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { QuizModal } from './QuizModal';
import { NotesPanel } from './NotesPanel';
import { QuizService } from '../services/quizService';
import { useProgress } from '../hooks/useProgress';
import type { Quiz } from '../data/quiz.types';
import './MarkdownViewer.css';
import './DocPage.css';

interface DocPageClientProps {
  docId: string;
  title: string;
  markdownContent: string;
  nextDocId?: string;
  nextDocTitle?: string;
}

export const DocPageClient = ({
  docId,
  title,
  markdownContent,
  nextDocId,
  nextDocTitle,
}: DocPageClientProps) => {
  const [isQuizOpen, setIsQuizOpen] = useState(false);
  const [isNotesOpen, setIsNotesOpen] = useState(false);
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [loadingQuiz, setLoadingQuiz] = useState(true);
  const { markAsVisited, updateTimeSpent, getChapterProgress, markAsCompleted } =
    useProgress();

  useEffect(() => {
    markAsVisited(docId);
    setLoadingQuiz(true);
    QuizService.loadQuiz(docId)
      .then((quizData) => {
        setQuiz(quizData);
      })
      .finally(() => {
        setLoadingQuiz(false);
      });
  }, [docId, markAsVisited]);

  useEffect(() => {
    return () => {
      updateTimeSpent(docId);
    };
  }, [docId, updateTimeSpent]);

  const progress = getChapterProgress(docId);

  const nextDoc =
    nextDocId && nextDocTitle
      ? { id: nextDocId, title: nextDocTitle, path: '' }
      : undefined;

  const handleQuizComplete = (passed: boolean) => {
    if (passed) {
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

      <article className="markdown-viewer">
        <h1 className="doc-title-bar">{title}</h1>
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {markdownContent}
          </ReactMarkdown>
        </div>
      </article>

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

      <NotesPanel
        chapterId={docId}
        isOpen={isNotesOpen}
        onClose={() => setIsNotesOpen(false)}
      />
    </>
  );
};
