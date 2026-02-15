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
import { absoluteUrl, SITE } from '@/config/site';
import { DocPageClient } from '@/components/DocPageClient';

interface PageProps {
  params: Promise<{ docId: string }>;
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
  const description = `${doc.title} - ${levelLabel}。AI Friends Schoolの教材。`;
  const canonicalPath = `/doc/${docId}`;

  return {
    title: doc.title,
    description,
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
      title: `${doc.title} | ${SITE.name}`,
      description,
      images: [`${canonicalPath}/twitter-image`],
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
  const contentPath = path.join(process.cwd(), 'content', 'docs', doc.path);
  let markdownContent = '';

  try {
    markdownContent = fs.readFileSync(contentPath, 'utf-8');
  } catch {
    notFound();
  }

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

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'LearningResource',
    '@id': `${canonicalUrl}#learning-resource`,
    name: doc.title,
    description,
    url: canonicalUrl,
    educationalLevel,
    inLanguage: 'ja',
    learningResourceType: 'lesson',
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
    itemListElement: [
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
