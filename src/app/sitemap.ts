import type { MetadataRoute } from 'next';
import fs from 'fs';
import path from 'path';
import { curriculum } from '@/data/curriculum';
import { SITE } from '@/config/site';

function safeMtime(filePath: string): Date {
  try {
    return fs.statSync(filePath).mtime;
  } catch {
    return new Date();
  }
}

export default function sitemap(): MetadataRoute.Sitemap {
  const contentRoot = path.join(process.cwd(), 'content', 'docs');
  const docPages = curriculum.flatMap((section) =>
    section.items.map((item) => {
      const contentPath = path.join(contentRoot, item.path);
      return {
        url: `${SITE.url}/doc/${item.id}`,
        lastModified: safeMtime(contentPath),
        changeFrequency: 'monthly' as const,
        priority: 0.8,
      };
    })
  );

  return [
    {
      url: SITE.url,
      lastModified: docPages.reduce<Date>(
        (latest, entry) =>
          entry.lastModified && entry.lastModified > latest
            ? entry.lastModified
            : latest,
        new Date(0)
      ),
      changeFrequency: 'weekly',
      priority: 1.0,
    },
    ...docPages,
  ];
}
