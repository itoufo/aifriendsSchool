import type { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { BASE_SEO_KEYWORDS } from '@/config/seo';
import { absoluteUrl, SITE } from '@/config/site';
import { getDocById } from '@/data/curriculum';
import {
  estimateNewsReadingMinutes,
  formatNewsDate,
  getAllNewsArticles,
  getNewsBySlug,
  getRelatedNewsArticles,
} from '@/data/news';
import styles from '../news.module.css';

interface PageProps {
  params: Promise<{ slug: string }>;
}

function toHeadingId(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\u3040-\u30ff\u4e00-\u9faf]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export async function generateStaticParams() {
  return getAllNewsArticles().map((article) => ({
    slug: article.slug,
  }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const article = getNewsBySlug(slug);
  if (!article) {
    return {
      robots: {
        index: false,
        follow: false,
      },
    };
  }

  const canonicalPath = `/news/${article.slug}`;
  const keywords = [...BASE_SEO_KEYWORDS, 'AIニュース', article.category, ...article.tags];
  const updatedAt = article.updatedAt || article.publishedAt;

  return {
    title: article.title,
    description: article.summary,
    keywords,
    alternates: {
      canonical: canonicalPath,
    },
    openGraph: {
      type: 'article',
      title: `${article.title} | ${SITE.name}`,
      description: article.summary,
      url: canonicalPath,
      siteName: SITE.name,
      locale: SITE.locale,
      publishedTime: new Date(article.publishedAt).toISOString(),
      modifiedTime: new Date(updatedAt).toISOString(),
      section: article.category,
      tags: keywords,
      images: [
        {
          url: `${canonicalPath}/opengraph-image`,
          width: 1200,
          height: 630,
          alt: article.title,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title: `${article.title} | ${SITE.name}`,
      description: article.summary,
      images: [`${canonicalPath}/opengraph-image`],
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

export default async function NewsArticlePage({ params }: PageProps) {
  const { slug } = await params;
  const article = getNewsBySlug(slug);
  if (!article) {
    notFound();
  }

  const canonicalUrl = absoluteUrl(`/news/${article.slug}`);
  const updatedAt = article.updatedAt || article.publishedAt;
  const readingMinutes = estimateNewsReadingMinutes(article);
  const relatedArticles = getRelatedNewsArticles(article.slug, 3);
  const relatedDocs = article.relatedDocIds
    .map((docId) => getDocById(docId))
    .filter((doc): doc is NonNullable<typeof doc> => Boolean(doc));

  const newsJsonLd = {
    '@context': 'https://schema.org',
    '@type': 'NewsArticle',
    '@id': `${canonicalUrl}#article`,
    mainEntityOfPage: canonicalUrl,
    headline: article.title,
    description: article.summary,
    image: [absoluteUrl(article.coverImage), absoluteUrl(`/news/${article.slug}/opengraph-image`)],
    datePublished: new Date(article.publishedAt).toISOString(),
    dateModified: new Date(updatedAt).toISOString(),
    timeRequired: `PT${readingMinutes}M`,
    articleSection: article.category,
    keywords: article.tags,
    about: article.tags.map((tag) => ({
      '@type': 'Thing',
      name: tag,
    })),
    inLanguage: SITE.language,
    isPartOf: {
      '@id': `${absoluteUrl('/news')}#collection`,
    },
    author: {
      '@id': `${SITE.url}/#org`,
    },
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
        name: 'AIニュース',
        item: absoluteUrl('/news'),
      },
      {
        '@type': 'ListItem',
        position: 3,
        name: article.title,
        item: canonicalUrl,
      },
    ],
  };

  return (
    <div className={styles.articlePage}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(newsJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />

      <div className={styles.articleContainer}>
        <nav className={styles.breadcrumb} aria-label="パンくずリスト">
          <Link href="/">ホーム</Link>
          <span>/</span>
          <Link href="/news">AIニュース</Link>
          <span>/</span>
          <span>{article.title}</span>
        </nav>

        <article className={styles.article}>
          <header>
            <div className={styles.metaRow}>
              <Link
                href={`/news/category/${article.categorySlug}`}
                className={styles.category}
              >
                {article.category}
              </Link>
              <time className={styles.date} dateTime={article.publishedAt}>
                公開: {formatNewsDate(article.publishedAt)}
              </time>
              {article.updatedAt ? (
                <time className={styles.date} dateTime={article.updatedAt}>
                  更新: {formatNewsDate(article.updatedAt)}
                </time>
              ) : null}
              <span className={styles.readingTime}>読了目安: {readingMinutes}分</span>
            </div>

            <h1 className={styles.headline}>{article.title}</h1>
            <p className={styles.articleSummary}>{article.summary}</p>

            <div className={styles.coverWrap}>
              <Image
                src={article.coverImage}
                alt={article.title}
                width={1408}
                height={768}
                className={styles.coverImage}
                priority
              />
            </div>
          </header>

          <div className={styles.sections}>
            <section className={styles.toc}>
              <h2 className={styles.tocTitle}>目次</h2>
              <ol className={styles.tocList}>
                {article.sections.map((section) => {
                  const headingId = toHeadingId(section.heading);
                  return (
                    <li key={headingId}>
                      <a href={`#${headingId}`}>{section.heading}</a>
                    </li>
                  );
                })}
              </ol>
            </section>

            {article.sections.map((section) => (
              <section key={section.heading} className={styles.section}>
                <h2 id={toHeadingId(section.heading)}>{section.heading}</h2>
                {section.paragraphs.map((paragraph) => (
                  <p key={paragraph}>{paragraph}</p>
                ))}
              </section>
            ))}

            {relatedDocs.length > 0 ? (
              <section className={styles.learningCta}>
                <h2 className={styles.learningCtaTitle}>このトピックを深掘りする教材</h2>
                <p>
                  ニュースの理解を実務スキルに変えるための推奨教材です。3本だけ先に進めると、
                  現場での再現性が上がります。
                </p>
                <div className={styles.learningDocLinks}>
                  {relatedDocs.map((doc) => (
                    <Link key={doc.id} href={`/doc/${doc.id}`}>
                      {doc.title}
                    </Link>
                  ))}
                </div>
              </section>
            ) : null}
          </div>

          <footer className={styles.articleFooter}>
            <div className={styles.tagList}>
              {article.tags.map((tag) => (
                <span key={tag} className={styles.tag}>
                  #{tag}
                </span>
              ))}
            </div>
            <div className={styles.footerLinks}>
              <Link href="/news">AIニュース一覧へ</Link>
              <Link href="/doc/getting-started">学習カリキュラムを見る</Link>
            </div>
          </footer>
        </article>

        {relatedArticles.length > 0 ? (
          <section className={styles.relatedSection}>
            <h2 className={styles.relatedTitle}>関連記事</h2>
            <div className={styles.relatedGrid}>
              {relatedArticles.map((related) => (
                <article key={related.slug} className={styles.relatedCard}>
                  <div className={styles.metaRow}>
                    <Link
                      href={`/news/category/${related.categorySlug}`}
                      className={styles.category}
                    >
                      {related.category}
                    </Link>
                    <time className={styles.date} dateTime={related.publishedAt}>
                      {formatNewsDate(related.publishedAt)}
                    </time>
                  </div>
                  <h3>
                    <Link href={`/news/${related.slug}`}>{related.title}</Link>
                  </h3>
                  <p>{related.summary}</p>
                  <Link href={`/news/${related.slug}`} className={styles.readMore}>
                    記事を読む
                  </Link>
                </article>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
