import type { Metadata } from 'next';
import Link from 'next/link';
import { BASE_SEO_KEYWORDS } from '@/config/seo';
import { absoluteUrl, SITE } from '@/config/site';
import { formatNewsDate, getAllNewsArticles, getAllNewsCategories } from '@/data/news';
import styles from './news.module.css';

export const metadata: Metadata = {
  title: 'AIニュース・実務解説',
  description:
    '生成AIの最新トレンド、業務活用、ガバナンス、導入実務をAI Friends School編集部が分かりやすく解説。',
  keywords: [...BASE_SEO_KEYWORDS, 'AIニュース', '生成AIトレンド', 'AI活用事例'],
  alternates: {
    canonical: '/news',
    types: {
      'application/rss+xml': absoluteUrl('/news/rss.xml'),
    },
  },
  openGraph: {
    type: 'website',
    title: `AIニュース | ${SITE.name}`,
    description:
      'AIニュースを実務に繋げる解説メディア。導入トレンド、活用ノウハウ、運用課題を体系的にキャッチアップ。',
    url: '/news',
    locale: SITE.locale,
    siteName: SITE.name,
    images: [
      {
        url: '/news/opengraph-image',
        width: 1200,
        height: 630,
        alt: 'AIニュース',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: `AIニュース | ${SITE.name}`,
    description:
      '生成AIの最新トレンドと実務ノウハウを毎週更新。コンテンツマーケにも活かせるニュース解説を掲載。',
    images: ['/news/opengraph-image'],
  },
};

export default function NewsPage() {
  const articles = getAllNewsArticles();
  const categories = getAllNewsCategories();

  const jsonLd = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'CollectionPage',
        '@id': `${absoluteUrl('/news')}#collection`,
        url: absoluteUrl('/news'),
        name: `${SITE.name} AIニュース`,
        description:
          '生成AIの最新トレンド、導入実務、ガバナンス、運用ノウハウを扱うニュース・解説ページ。',
        inLanguage: SITE.language,
        isPartOf: {
          '@id': `${SITE.url}/#website`,
        },
      },
      {
        '@type': 'ItemList',
        '@id': `${absoluteUrl('/news')}#item-list`,
        name: 'AIニュース記事一覧',
        numberOfItems: articles.length,
        itemListElement: articles.map((article, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          name: article.title,
          url: absoluteUrl(`/news/${article.slug}`),
        })),
      },
      {
        '@type': 'ItemList',
        '@id': `${absoluteUrl('/news')}#categories`,
        name: 'AIニュースカテゴリ',
        numberOfItems: categories.length,
        itemListElement: categories.map((category, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          name: category.name,
          url: absoluteUrl(`/news/category/${category.slug}`),
        })),
      },
      ...articles.map((article) => ({
        '@type': 'NewsArticle',
        '@id': `${absoluteUrl(`/news/${article.slug}`)}#article`,
        headline: article.title,
        description: article.summary,
        datePublished: new Date(article.publishedAt).toISOString(),
        dateModified: new Date(article.updatedAt || article.publishedAt).toISOString(),
        articleSection: article.category,
        inLanguage: SITE.language,
        keywords: article.tags,
        mainEntityOfPage: absoluteUrl(`/news/${article.slug}`),
        image: [absoluteUrl(article.coverImage)],
        author: {
          '@id': `${SITE.url}/#org`,
        },
        publisher: {
          '@id': `${SITE.url}/#org`,
        },
      })),
    ],
  };

  return (
    <div className={styles.page}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <div className={styles.container}>
        <section className={styles.hero}>
          <h1 className={styles.title}>AIニュース・実務解説</h1>
          <p className={styles.description}>
            教材だけで終わらない、実務に直結するAIメディア。最新動向を
            「何が変わるか」「どう使うか」の観点で整理します。
          </p>
          <div className={styles.actions}>
            <Link href="/news/rss.xml" className={styles.actionLink}>
              RSSを購読
            </Link>
            <Link href="/" className={styles.actionLink}>
              教材トップへ戻る
            </Link>
          </div>
        </section>

        <section className={styles.categorySection}>
          <h2 className={styles.sectionTitle}>カテゴリ</h2>
          <div className={styles.categoryLinks}>
            {categories.map((category) => (
              <Link
                key={category.slug}
                href={`/news/category/${category.slug}`}
                className={styles.categoryLink}
              >
                {category.name}
                <span>{category.count}</span>
              </Link>
            ))}
          </div>
        </section>

        <section className={styles.list}>
          {articles.map((article) => (
            <article key={article.slug} className={styles.card}>
              <div className={styles.metaRow}>
                <Link
                  href={`/news/category/${article.categorySlug}`}
                  className={styles.category}
                >
                  {article.category}
                </Link>
                <time className={styles.date} dateTime={article.publishedAt}>
                  {formatNewsDate(article.publishedAt)}
                  {article.updatedAt ? ` 更新: ${formatNewsDate(article.updatedAt)}` : ''}
                </time>
              </div>

              <h2 className={styles.cardTitle}>
                <Link href={`/news/${article.slug}`}>{article.title}</Link>
              </h2>
              <p className={styles.summary}>{article.summary}</p>

              <div className={styles.tagList}>
                {article.tags.map((tag) => (
                  <span key={tag} className={styles.tag}>
                    #{tag}
                  </span>
                ))}
              </div>

              <Link href={`/news/${article.slug}`} className={styles.readMore}>
                記事を読む
              </Link>
            </article>
          ))}
        </section>
      </div>
    </div>
  );
}
