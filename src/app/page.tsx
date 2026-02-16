import type { Metadata } from 'next';
import { HomePage } from '@/components/HomePage';
import { SEO_FAQ_ITEMS, BASE_SEO_KEYWORDS } from '@/config/seo';
import { absoluteUrl, SITE } from '@/config/site';
import { curriculum } from '@/data/curriculum';
import { getLatestNews } from '@/data/news';

export const metadata: Metadata = {
  title: 'AI活用を体系的に学ぶオンラインスクール',
  description:
    'AI初心者から経営者まで対応。5レベルの教材に加えて、AIニュース・実務解説も配信する学習メディア。',
  keywords: [
    ...BASE_SEO_KEYWORDS,
    'AIオンラインスクール',
    'AIカリキュラム',
    'AIニュースメディア',
    'コンテンツマーケ',
  ],
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    title: SITE.title,
    description: SITE.description,
    url: '/',
    locale: SITE.locale,
    siteName: SITE.name,
    images: [
      {
        url: '/opengraph-image',
        width: 1200,
        height: 630,
        alt: SITE.title,
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: SITE.title,
    description: SITE.description,
    images: ['/twitter-image'],
  },
};

export default function Page() {
  const lessonItems = curriculum.flatMap((section) =>
    section.items.map((item) => ({
      item,
      section,
    }))
  );
  const latestNews = getLatestNews(5);

  const jsonLdGraph = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Organization',
        '@id': `${SITE.url}/#org`,
        name: SITE.name,
        description: SITE.description,
        url: SITE.url,
        logo: absoluteUrl('/images/logo.png'),
        inLanguage: SITE.language,
      },
      {
        '@type': 'WebSite',
        '@id': `${SITE.url}/#website`,
        url: SITE.url,
        name: SITE.name,
        description: SITE.description,
        inLanguage: SITE.language,
        publisher: {
          '@id': `${SITE.url}/#org`,
        },
      },
      {
        '@type': 'Course',
        '@id': `${SITE.url}/#course`,
        name: SITE.title,
        description: SITE.description,
        inLanguage: SITE.language,
        provider: {
          '@id': `${SITE.url}/#org`,
        },
        hasPart: curriculum.map((section) => ({
          '@type': 'Course',
          '@id': `${SITE.url}/#course-${section.id}`,
          name: section.title,
          description: section.description,
          inLanguage: SITE.language,
          provider: {
            '@id': `${SITE.url}/#org`,
          },
        })),
      },
      {
        '@type': 'ItemList',
        '@id': `${SITE.url}/#curriculum`,
        name: 'AI Friends School カリキュラム一覧',
        numberOfItems: lessonItems.length,
        itemListElement: lessonItems.map(({ item }, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          name: item.title,
          url: absoluteUrl(`/doc/${item.id}`),
        })),
      },
      {
        '@type': 'Blog',
        '@id': `${SITE.url}/news#blog`,
        name: `${SITE.name} AIニュース`,
        url: absoluteUrl('/news'),
        description:
          '生成AIの最新トレンド、導入実務、コンテンツマーケに役立つニュース解説を配信。',
        inLanguage: SITE.language,
        publisher: {
          '@id': `${SITE.url}/#org`,
        },
      },
      {
        '@type': 'ItemList',
        '@id': `${SITE.url}/#latest-news`,
        name: '最新AIニュース',
        numberOfItems: latestNews.length,
        itemListElement: latestNews.map((article, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          name: article.title,
          url: absoluteUrl(`/news/${article.slug}`),
        })),
      },
      {
        '@type': 'FAQPage',
        '@id': `${SITE.url}/#faq`,
        mainEntity: SEO_FAQ_ITEMS.map((faq) => ({
          '@type': 'Question',
          name: faq.question,
          acceptedAnswer: {
            '@type': 'Answer',
            text: faq.answer,
          },
        })),
      },
    ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLdGraph),
        }}
      />
      <HomePage />
    </>
  );
}
