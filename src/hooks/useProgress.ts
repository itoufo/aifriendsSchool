import { useState, useEffect, useCallback } from 'react';

export interface ChapterProgress {
  chapterId: string;
  completed: boolean;
  completedAt?: string;
  timeSpent: number;
  lastVisited?: string;
}

export interface LearningStats {
  totalChapters: number;
  completedChapters: number;
  totalTimeSpent: number;
  currentStreak: number;
  longestStreak: number;
  lastActivityDate?: string;
}

const STORAGE_KEY = 'ai-school-progress';
const STATS_KEY = 'ai-school-stats';

export const useProgress = () => {
  const [progress, setProgress] = useState<Map<string, ChapterProgress>>(new Map());
  const [stats, setStats] = useState<LearningStats>({
    totalChapters: 0,
    completedChapters: 0,
    totalTimeSpent: 0,
    currentStreak: 0,
    longestStreak: 0,
  });
  const [sessionStartTime, setSessionStartTime] = useState<number>(Date.now());

  useEffect(() => {
    const loadedProgress = localStorage.getItem(STORAGE_KEY);
    if (loadedProgress) {
      const parsed = JSON.parse(loadedProgress);
      setProgress(new Map(Object.entries(parsed)));
    }

    const loadedStats = localStorage.getItem(STATS_KEY);
    if (loadedStats) {
      setStats(JSON.parse(loadedStats));
    }
  }, []);

  useEffect(() => {
    if (progress.size > 0) {
      const progressObj = Object.fromEntries(progress);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(progressObj));
      
      const completed = Array.from(progress.values()).filter(p => p.completed).length;
      setStats(prev => ({
        ...prev,
        completedChapters: completed,
        totalChapters: progress.size,
      }));
    }
  }, [progress]);

  useEffect(() => {
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
  }, [stats]);

  const markAsCompleted = useCallback((chapterId: string) => {
    setProgress(prev => {
      const newProgress = new Map(prev);
      const existing = newProgress.get(chapterId) || {
        chapterId,
        completed: false,
        timeSpent: 0,
      };

      newProgress.set(chapterId, {
        ...existing,
        completed: true,
        completedAt: new Date().toISOString(),
      });

      return newProgress;
    });

    setStats(prev => ({
      ...prev,
      completedChapters: prev.completedChapters + 1,
      lastActivityDate: new Date().toISOString(),
    }));
  }, []);

  const markAsVisited = useCallback((chapterId: string) => {
    setSessionStartTime(Date.now());
    
    setProgress(prev => {
      const newProgress = new Map(prev);
      const existing = newProgress.get(chapterId) || {
        chapterId,
        completed: false,
        timeSpent: 0,
      };

      newProgress.set(chapterId, {
        ...existing,
        lastVisited: new Date().toISOString(),
      });

      return newProgress;
    });
  }, []);

  const updateTimeSpent = useCallback((chapterId: string) => {
    const timeSpent = Math.floor((Date.now() - sessionStartTime) / 1000);
    
    setProgress(prev => {
      const newProgress = new Map(prev);
      const existing = newProgress.get(chapterId) || {
        chapterId,
        completed: false,
        timeSpent: 0,
      };

      newProgress.set(chapterId, {
        ...existing,
        timeSpent: existing.timeSpent + timeSpent,
      });

      return newProgress;
    });

    setStats(prev => ({
      ...prev,
      totalTimeSpent: prev.totalTimeSpent + timeSpent,
    }));
  }, [sessionStartTime]);

  const getChapterProgress = useCallback((chapterId: string): ChapterProgress | undefined => {
    return progress.get(chapterId);
  }, [progress]);

  const getCompletionPercentage = useCallback((): number => {
    if (stats.totalChapters === 0) return 0;
    return Math.round((stats.completedChapters / stats.totalChapters) * 100);
  }, [stats]);

  const resetProgress = useCallback(() => {
    setProgress(new Map());
    setStats({
      totalChapters: 0,
      completedChapters: 0,
      totalTimeSpent: 0,
      currentStreak: 0,
      longestStreak: 0,
    });
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STATS_KEY);
  }, []);

  const exportProgress = useCallback(() => {
    const data = {
      progress: Object.fromEntries(progress),
      stats,
      exportedAt: new Date().toISOString(),
    };
    return JSON.stringify(data, null, 2);
  }, [progress, stats]);

  const importProgress = useCallback((jsonData: string) => {
    try {
      const data = JSON.parse(jsonData);
      if (data.progress) {
        setProgress(new Map(Object.entries(data.progress)));
      }
      if (data.stats) {
        setStats(data.stats);
      }
    } catch (error) {
      console.error('Failed to import progress:', error);
      throw new Error('Invalid progress data format');
    }
  }, []);

  return {
    progress,
    stats,
    markAsCompleted,
    markAsVisited,
    updateTimeSpent,
    getChapterProgress,
    getCompletionPercentage,
    resetProgress,
    exportProgress,
    importProgress,
  };
};