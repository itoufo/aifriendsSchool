export interface DocItem {
  id: string;
  title: string;
  path: string;
}

export interface Section {
  id: string;
  title: string;
  items: DocItem[];
}

export const curriculum: Section[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    items: [
      { id: 'introduction', title: 'Introduction', path: '/docs/01_getting-started/introduction.md' },
      { id: 'quick-start', title: 'Quick Start Guide', path: '/docs/01_getting-started/quick-start.md' },
      { id: 'project-structure', title: 'Project Structure', path: '/docs/01_getting-started/project-structure.md' },
    ],
  },
  {
    id: 'basic-usage',
    title: 'Basic Usage',
    items: [
      { id: 'creating-content', title: 'Creating Content', path: '/docs/02_basic-usage/creating-content.md' },
      { id: 'navigation-setup', title: 'Navigation Setup', path: '/docs/02_basic-usage/navigation-setup.md' },
      { id: 'quiz-system', title: 'Quiz System', path: '/docs/02_basic-usage/quiz-system.md' },
    ],
  },
  {
    id: 'advanced-features',
    title: 'Advanced Features',
    items: [
      { id: 'progress-tracking', title: 'Progress Tracking', path: '/docs/03_advanced-features/progress-tracking.md' },
      { id: 'notes-bookmarks', title: 'Notes & Bookmarks', path: '/docs/03_advanced-features/notes-bookmarks.md' },
      { id: 'customization', title: 'Customization', path: '/docs/03_advanced-features/customization.md' },
    ],
  },
  {
    id: 'deployment',
    title: 'Deployment',
    items: [
      { id: 'building', title: 'Building for Production', path: '/docs/04_deployment/building.md' },
      { id: 'hosting', title: 'Hosting Options', path: '/docs/04_deployment/hosting.md' },
    ],
  },
];

export const getDocByPath = (path: string): DocItem | undefined => {
  for (const section of curriculum) {
    const item = section.items.find((item) => item.path === path);
    if (item) return item;
  }
  return undefined;
};

export const getDocById = (id: string): DocItem | undefined => {
  for (const section of curriculum) {
    const item = section.items.find((item) => item.id === id);
    if (item) return item;
  }
  return undefined;
};

/**
 * 指定された章IDの次の章を取得
 */
export const getNextDoc = (currentId: string): DocItem | undefined => {
  const allDocs: DocItem[] = [];
  for (const section of curriculum) {
    allDocs.push(...section.items);
  }

  const currentIndex = allDocs.findIndex((doc) => doc.id === currentId);
  if (currentIndex === -1 || currentIndex === allDocs.length - 1) {
    return undefined;
  }

  return allDocs[currentIndex + 1];
};
