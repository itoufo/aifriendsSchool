# Progress Tracking

Learn how the framework automatically tracks user progress and completion.

## Overview

The progress tracking system provides:
- 📊 **Automatic progress monitoring** based on quiz completion
- ⏱️ **Time tracking** for each chapter  
- 🎯 **Completion status** with visual indicators
- 💾 **Persistent storage** in browser localStorage
- 📈 **Progress visualization** in the sidebar

## How Progress Tracking Works

### Completion Criteria
A chapter is marked as "completed" when:
1. User takes the associated quiz
2. Achieves a passing score (80% by default)
3. Quiz completion is automatically saved

### Progress Data Structure
```typescript
interface ChapterProgress {
  chapterId: string;        // Document ID
  completed: boolean;       // True if quiz passed
  completedAt?: string;     // ISO timestamp of completion
  timeSpent: number;        // Total time spent in milliseconds
  lastVisited?: string;     // Last visit timestamp
  quizPassed: boolean;      // Quiz pass status
  quizScore?: number;       // Quiz score (0-1)
  quizAttempts?: number;    // Number of quiz attempts
}
```

## Visual Indicators

### Sidebar Progress Icons
- ✅ **Green checkmark**: Chapter completed (quiz passed)
- 📖 **Blue book**: Currently reading
- ⭕ **Gray circle**: Not yet completed
- 🔄 **Yellow refresh**: In progress (visited but quiz not passed)

### Progress Bar
A progress bar shows overall completion:
```
Progress: ████████░░ 80% (4/5 chapters)
```

## Using the Progress Hook

### Basic Usage
```typescript
import { useProgress } from '../hooks/useProgress';

function MyComponent() {
  const { 
    progress,           // All progress data
    updateProgress,     // Function to update progress
    getProgress,        // Get progress for specific chapter
    getTotalProgress    // Get overall progress stats
  } = useProgress();

  // Get progress for a specific chapter
  const chapterProgress = getProgress('introduction');
  console.log(chapterProgress?.completed); // true/false

  // Get overall stats
  const stats = getTotalProgress();
  console.log(stats.completionRate); // 0.8 (80%)
}
```

### Manual Progress Updates
```typescript
// Mark a chapter as visited (starts time tracking)
updateProgress('chapter-id', { 
  lastVisited: new Date().toISOString(),
  timeSpent: existingTime + newTime
});

// Mark quiz as completed
updateProgress('chapter-id', {
  completed: true,
  completedAt: new Date().toISOString(),
  quizPassed: true,
  quizScore: 0.9,
  quizAttempts: 1
});
```

## Progress Analytics

### Time Tracking
Time spent is automatically tracked:

```typescript
const stats = getTotalProgress();
console.log(stats.totalTimeSpent);    // Total milliseconds across all chapters
console.log(stats.averageTimePerChapter); // Average time per chapter
```

### Completion Analytics  
```typescript
const stats = getTotalProgress();
console.log(stats.completedChapters);  // Number of completed chapters
console.log(stats.totalChapters);      // Total number of chapters  
console.log(stats.completionRate);     // Decimal completion rate (0-1)
```

### Chapter-Specific Data
```typescript
const chapterProgress = getProgress('advanced-config');
if (chapterProgress) {
  console.log(`Completed: ${chapterProgress.completed}`);
  console.log(`Time spent: ${chapterProgress.timeSpent / 1000 / 60} minutes`);
  console.log(`Quiz score: ${chapterProgress.quizScore * 100}%`);
  console.log(`Attempts: ${chapterProgress.quizAttempts}`);
}
```

## Customizing Progress Tracking

### Changing Pass Threshold
Modify the quiz pass threshold:

```typescript
// In src/services/quizService.ts
const QUIZ_PASS_THRESHOLD = 0.7; // 70% instead of 80%
```

### Custom Completion Criteria
Override the default completion logic:

```typescript
// In src/hooks/useProgress.ts
const isChapterComplete = (progress: ChapterProgress): boolean => {
  // Custom logic: require both quiz pass AND minimum time spent
  return progress.quizPassed && progress.timeSpent > 300000; // 5 minutes
};
```

### Additional Tracking Metrics
Extend the progress interface:

```typescript
interface ExtendedProgress extends ChapterProgress {
  bookmarked?: boolean;    // User bookmarked this chapter
  noteCount?: number;      // Number of notes taken
  revisits?: number;       // How many times revisited
}
```

## Progress Export and Import

### Export Progress Data
```typescript
const exportProgress = () => {
  const progressData = localStorage.getItem('knowledgeFramework_progress');
  const blob = new Blob([progressData], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = 'learning-progress.json';
  a.click();
};
```

### Import Progress Data
```typescript
const importProgress = (file: File) => {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target?.result as string);
      localStorage.setItem('knowledgeFramework_progress', JSON.stringify(data));
      window.location.reload(); // Refresh to load new data
    } catch (error) {
      console.error('Invalid progress file');
    }
  };
  reader.readAsText(file);
};
```

## Data Storage and Privacy

### Local Storage Schema
```
localStorage['knowledgeFramework_progress'] = {
  "introduction": {
    "chapterId": "introduction",
    "completed": true,
    "completedAt": "2024-01-15T10:30:00Z",
    "timeSpent": 480000,
    "quizPassed": true,
    "quizScore": 0.9,
    "quizAttempts": 1
  },
  // ... more chapters
}
```

### Privacy Considerations
- All data stored locally in browser
- No data sent to external servers
- Users can clear data anytime via browser settings
- Export functionality allows backup

### Data Persistence
- Data persists across browser sessions
- Survives page refreshes and tab closing
- Lost only if user clears browser storage
- Consider adding backup/restore functionality

## Troubleshooting

### Progress Not Saving
1. Check browser localStorage is enabled
2. Verify browser storage quota not exceeded
3. Ensure no JavaScript errors in console
4. Test in incognito mode to isolate extensions

### Progress Not Displaying
1. Check localStorage contains progress data
2. Verify useProgress hook is properly imported
3. Ensure curriculum IDs match progress keys
4. Check browser console for React errors

### Performance Issues
1. Monitor localStorage size (2MB typical limit)
2. Consider implementing data cleanup for old entries
3. Use pagination for very large progress datasets

## Integration Examples

### Custom Progress Dashboard
```tsx
function ProgressDashboard() {
  const { getTotalProgress } = useProgress();
  const stats = getTotalProgress();
  
  return (
    <div className="progress-dashboard">
      <h2>Your Learning Progress</h2>
      <div className="stats-grid">
        <div className="stat">
          <h3>{stats.completedChapters}</h3>
          <p>Chapters Completed</p>
        </div>
        <div className="stat">
          <h3>{Math.round(stats.completionRate * 100)}%</h3>
          <p>Overall Progress</p>
        </div>
        <div className="stat">
          <h3>{Math.round(stats.totalTimeSpent / 1000 / 60)}</h3>
          <p>Minutes Spent Learning</p>
        </div>
      </div>
    </div>
  );
}
```

### Achievement System
```typescript
const getAchievements = (progress: Record<string, ChapterProgress>) => {
  const achievements = [];
  
  // First completion
  if (Object.values(progress).some(p => p.completed)) {
    achievements.push('🎉 First Chapter Complete!');
  }
  
  // Perfect scores
  const perfectScores = Object.values(progress).filter(p => p.quizScore === 1).length;
  if (perfectScores >= 3) {
    achievements.push('🎯 Quiz Master - 3 Perfect Scores!');
  }
  
  // Speed learner
  const fastCompletions = Object.values(progress)
    .filter(p => p.completed && p.timeSpent < 300000) // Under 5 minutes
    .length;
  if (fastCompletions >= 2) {
    achievements.push('⚡ Speed Learner!');
  }
  
  return achievements;
};
```

## Next Steps

- Set up [notes and bookmarks](notes-bookmarks.md) for enhanced learning
- Explore [customization options](customization.md) to tailor the framework
- Learn about [deployment strategies](../04_deployment/building.md)