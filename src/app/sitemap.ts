import type { MetadataRoute } from 'next';
import fs from 'fs';
import path from 'path';
import { curriculum } from '@/data/curriculum';
import { getAllNewsArticles, getAllNewsCategories, getNewsByCategorySlug } from '@/data/news';
import { absoluteUrl, SITE } from '@/config/site';

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
        url: absoluteUrl(`/doc/${item.id}`),
        lastModified: safeMtime(contentPath),
        changeFrequency: 'monthly' as const,
        priority: 0.8,
        images: [absoluteUrl(`/doc/${item.id}/opengraph-image`)],
      };
    })
  );
  const newsArticles = getAllNewsArticles();
  const newsPages = newsArticles.map((article) => ({
    url: absoluteUrl(`/news/${article.slug}`),
    lastModified: new Date(article.updatedAt || article.publishedAt),
    changeFrequency: 'weekly' as const,
    priority: 0.7,
    images: [absoluteUrl(`/news/${article.slug}/opengraph-image`)],
  }));
  const categoryPages = getAllNewsCategories().map((category) => {
    const articles = getNewsByCategorySlug(category.slug);
    const lastModified = articles.reduce<Date>(
      (latest, article) => {
        const articleDate = new Date(article.updatedAt || article.publishedAt);
        return articleDate > latest ? articleDate : latest;
      },
      new Date(0)
    );

    return {
      url: absoluteUrl(`/news/category/${category.slug}`),
      lastModified,
      changeFrequency: 'weekly' as const,
      priority: 0.6,
      images: [absoluteUrl('/news/opengraph-image')],
    };
  });

  const latestDocModified = docPages.reduce<Date>(
    (latest, entry) =>
      entry.lastModified && entry.lastModified > latest
        ? entry.lastModified
        : latest,
    new Date(0)
  );
  const latestNewsModified =
    newsPages.length > 0
      ? newsPages.reduce<Date>(
          (latest, entry) =>
            entry.lastModified && entry.lastModified > latest
              ? entry.lastModified
              : latest,
          new Date(0)
        )
      : latestDocModified;
  const latestSiteModified =
    latestNewsModified > latestDocModified ? latestNewsModified : latestDocModified;

  return [
    {
      url: SITE.url,
      lastModified: latestSiteModified,
      changeFrequency: 'weekly',
      priority: 1.0,
      images: [absoluteUrl('/opengraph-image'), absoluteUrl('/images/logo.png')],
    },
    {
      url: absoluteUrl('/news'),
      lastModified: latestNewsModified,
      changeFrequency: 'daily',
      priority: 0.9,
      images: [absoluteUrl('/news/opengraph-image')],
    },
    ...categoryPages,
    ...docPages,
    ...newsPages,
  ];
}
