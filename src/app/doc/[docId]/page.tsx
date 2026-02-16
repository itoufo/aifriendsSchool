import type { Metadata } from 'next';
import fs from 'fs';
import path from 'path';
import { notFound } from 'next/navigation';
import {
  curriculum,
  getDocById,
  getNextDoc,
  getSectionByDocId,
} from '@/data/curriculum';
import { buildDocKeywords, extractMarkdownSummary } from '@/config/seo';
import { absoluteUrl, SITE } from '@/config/site';
import { DocPageClient } from '@/components/DocPageClient';

interface PageProps {
  params: Promise<{ docId: string }>;
}

function getContentPath(docPath: string): string {
  return path.join(process.cwd(), 'content', 'docs', docPath);
}

function readDocContent(contentPath: string): string | null {
  try {
    return fs.readFileSync(contentPath, 'utf-8');
  } catch {
    return null;
  }
}

function readDocDates(contentPath: string): {
  publishedTime?: string;
  modifiedTime?: string;
} {
  try {
    const stats = fs.statSync(contentPath);
    return {
      publishedTime: stats.birthtime.toISOString(),
      modifiedTime: stats.mtime.toISOString(),
    };
  } catch {
    return {};
  }
}

export async function generateStaticParams() {
  const allDocs = curriculum.flatMap((section) =>
    section.items.map((item) => ({ docId: item.id }))
  );
  return allDocs;
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { docId } = await params;
  const doc = getDocById(docId);
  if (!doc) return {};

  const section = getSectionByDocId(docId);
  const levelLabel = section ? section.title : '';
  const contentPath = getContentPath(doc.path);
  const fallbackDescription = `${doc.title}${levelLabel ? ` - ${levelLabel}` : ''}。AI Friends Schoolの教材。`;
  const markdown = readDocContent(contentPath) || '';
  const description = extractMarkdownSummary(markdown) || fallbackDescription;
  const { publishedTime, modifiedTime } = readDocDates(contentPath);
  const canonicalPath = `/doc/${docId}`;
  const keywords = buildDocKeywords(doc, section);

  return {
    title: doc.title,
    description,
    keywords,
    category: 'education',
    alternates: {
      canonical: canonicalPath,
    },
    openGraph: {
      title: `${doc.title} | ${SITE.name}`,
      description,
      url: canonicalPath,
      type: 'article',
      locale: SITE.locale,
      siteName: SITE.name,
      ...(publishedTime ? { publishedTime } : {}),
      ...(modifiedTime ? { modifiedTime } : {}),
      ...(section ? { section: section.title } : {}),
      tags: keywords,
      images: [
        {
          url: `${canonicalPath}/opengraph-image`,
          width: 1200,
          height: 630,
          alt: doc.title,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title: `${doc.title} | ${SITE.name}`,
      description,
      images: [`${canonicalPath}/twitter-image`],
    },
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        'max-snippet': -1,
        'max-image-preview': 'large',
        'max-video-preview': -1,
      },
    },
  };
}

export default async function DocPage({ params }: PageProps) {
  const { docId } = await params;
  const doc = getDocById(docId);

  if (!doc) {
    notFound();
  }

  const section = getSectionByDocId(docId);
  const nextDoc = getNextDoc(docId);

  // サーバーサイドでMarkdownを読み込む
  const contentPath = getContentPath(doc.path);
  const rawMarkdown = readDocContent(contentPath);
  if (!rawMarkdown) {
    notFound();
  }
  let markdownContent = rawMarkdown;

  // Avoid duplicate H1: the page title is rendered separately.
  {
    const normalized = markdownContent.replace(/\r\n/g, '\n');
    const escapedTitle = doc.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    markdownContent = normalized.replace(
      new RegExp(`^#\\s+${escapedTitle}\\s*\\n+`),
      ''
    );
  }

  const educationalLevel =
    doc.level === 'start'
      ? 'beginner'
      : doc.level === 'career'
        ? 'intermediate'
        : doc.level || 'beginner';

  const canonicalUrl = absoluteUrl(`/doc/${docId}`);
  const description = `${doc.title}${section ? ` - ${section.title}` : ''}。AI Friends Schoolの教材。`;
  const { modifiedTime } = readDocDates(contentPath);
  const keywords = buildDocKeywords(doc, section);

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'LearningResource',
    '@id': `${canonicalUrl}#learning-resource`,
    name: doc.title,
    description,
    url: canonicalUrl,
    image: absoluteUrl(`/doc/${docId}/opengraph-image`),
    educationalLevel,
    inLanguage: 'ja',
    learningResourceType: 'lesson',
    keywords,
    ...(modifiedTime ? { dateModified: modifiedTime } : {}),
    ...(section && {
      isPartOf: {
        '@type': 'Course',
        '@id': `${SITE.url}/#course-${section.id}`,
        name: section.title,
        description: section.description,
        provider: {
          '@id': `${SITE.url}/#org`,
        },
      },
    }),
    publisher: {
      '@id': `${SITE.url}/#org`,
    },
  };

  const breadcrumbJsonLd = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: section
      ? [
          {
            '@type': 'ListItem',
            position: 1,
            name: 'ホーム',
            item: SITE.url,
          },
          {
            '@type': 'ListItem',
            position: 2,
            name: section.title,
            item: absoluteUrl(`/#${section.id}`),
          },
          {
            '@type': 'ListItem',
            position: 3,
            name: doc.title,
            item: canonicalUrl,
          },
        ]
      : [
          {
            '@type': 'ListItem',
            position: 1,
            name: 'ホーム',
            item: SITE.url,
          },
          {
            '@type': 'ListItem',
            position: 2,
            name: doc.title,
            item: canonicalUrl,
          },
        ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <DocPageClient
        docId={docId}
        title={doc.title}
        markdownContent={markdownContent}
        nextDocId={nextDoc?.id}
        nextDocTitle={nextDoc?.title}
      />
    </>
  );
}
